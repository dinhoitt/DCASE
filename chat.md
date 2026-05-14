

네, **방향은 맞습니다.** 지금 수정하신 코드는 “BemaGANv2의 MED를 DCASE용 feature extractor로 전환한다”는 목적에 상당히 잘 맞습니다. 특히 기존 `Envelope(...).envelope` 방식에서 벗어나 `FixedEnvelope(nn.Module)`로 바꾸고, two-channel 입력을 `near / far / diff`로 나눠 feature를 뽑는 구조는 적절합니다. 

다만 그대로 실험을 돌리기 전에 **몇 가지 수정이 꼭 필요해 보입니다.**

---

## 1. 현재 구조에서 잘한 부분

### 1.1 `FixedEnvelope`로 바꾼 것은 좋습니다

기존 `envelope.py`는 `.item()`과 `torch.tensor(...)` 재생성 때문에 gradient graph가 끊기고, batch 처리도 불안정했습니다. 지금은 `torch.fft` 기반으로 analytic signal과 low-pass envelope를 계산하도록 바꾸셨습니다. `FixedEnvelope.forward()`도 `max_freq=0`, `-1`, `1`, 그리고 low-pass envelope를 분기 처리하고 있어서 기존 MED 개념을 잘 살렸습니다. 

### 1.2 `DiscriminatorEForASD` 구조도 맞습니다

기존 MED discriminator의 convolution stack을 유지하되, `y`와 `y_hat`을 비교하지 않고 입력 waveform 하나에서 feature map을 반환하도록 바꾼 점이 맞습니다. 현재 `forward()`는 각 convolution 전에 envelope를 적용하고, feature map을 `fmap`에 저장한 뒤 flatten output과 feature map을 반환합니다. 

### 1.3 `near / far / diff` 입력은 DCASE 2026에 잘 맞습니다

`MEDFeatureExtractor.forward()`에서 입력이 `[B, C, T]`일 때 `near`, `far`, `diff = near - far`를 만들고 각각 feature를 추출한 뒤 concat하는 구조는 좋습니다. DCASE 2026의 two-channel noise-aware 설정에 맞는 설계입니다. 

### 1.4 waveform loader도 기본 방향은 맞습니다

`DCASE202XT2WaveformLoader`에서 `soundfile`로 always_2d로 읽고, `[channel, time]` 형태로 transpose한 뒤, mono면 2채널로 repeat하고, 2채널 초과면 앞 2채널만 쓰는 구조는 DCASE 2026용으로 적절합니다. resampling, padding/crop, peak normalization도 포함되어 있습니다. 

---

## 2. 반드시 고쳐야 할 부분 1: embedding 차원이 너무 작습니다

현재 `pool_fmaps()`는 다음처럼 되어 있습니다.

```python
feat.mean(dim=(1, 2))
feat.std(dim=(1, 2), unbiased=False)
feat.amin(dim=(1, 2))
feat.amax(dim=(1, 2))
```

이렇게 하면 각 feature map마다 **스칼라 4개**만 나옵니다. 즉, channel 정보가 모두 사라집니다. MED branch가 가진 128, 256, 512, 1024 채널 feature의 의미가 거의 날아갑니다. 

DCASE anomaly detection에서는 embedding이 너무 작으면 표현력이 부족할 수 있습니다. 저는 최소한 channel-wise pooling을 추천드립니다.

수정 전:

```python
feat.mean(dim=(1, 2))
```

수정 후:

```python
feat.mean(dim=-1)  # [B, C]
feat.std(dim=-1)   # [B, C]
```

추천 수정안은 다음입니다.

```python
@staticmethod
def pool_fmaps(fmaps):
    pooled = []
    for feat in fmaps:
        pooled.append(feat.mean(dim=-1))
        pooled.append(feat.std(dim=-1, unbiased=False))
    return torch.cat(pooled, dim=1)
```

이렇게 하면 feature map의 channel 구조를 보존합니다. 다만 embedding dimension이 커지므로 Mahalanobis covariance가 불안정해질 수 있습니다. 그래서 뒤에서 PCA 또는 diagonal covariance를 같이 쓰는 것이 좋습니다.

---

## 3. 반드시 고쳐야 할 부분 2: Mahalanobis가 고차원에서 불안정할 가능성이 큽니다

현재 `fit_gaussian()`은 embedding 전체 차원에 대해 full covariance를 만들고 `torch.linalg.pinv()`로 inverse covariance를 구합니다. 

문제는 embedding 차원이 커지면, training sample 수보다 feature dimension이 훨씬 커질 수 있다는 점입니다. 예를 들어 channel-wise pooling으로 바꾸면 feature dimension이 수천~수만 차원이 될 수 있습니다. 이 경우 covariance가 rank-deficient가 되고, Mahalanobis score가 불안정해질 가능성이 큽니다.

따라서 두 가지 중 하나를 추천합니다.

### 선택 A: 지금처럼 작은 embedding 유지

현재처럼 feature map당 스칼라 4개만 쓰면 covariance는 안정적입니다. 하지만 표현력이 약합니다.

### 선택 B: channel-wise pooling + PCA

이쪽을 더 추천합니다.

```text
MED feature extraction
→ StandardScaler
→ PCA(64 또는 128차원)
→ source/target Mahalanobis
```

DCASE용 첫 실험에서는 PCA 64차원이 안전합니다.

### 선택 C: diagonal Mahalanobis

full covariance 대신 variance만 쓰는 방식입니다.

[
D(z) = \sum_i \frac{(z_i - \mu_i)^2}{\sigma_i^2 + \epsilon}
]

샘플 수가 적은 target domain에서는 diagonal 방식이 오히려 더 안정적일 수 있습니다.

---

## 4. 반드시 고쳐야 할 부분 3: target/source 판별을 basename 문자열에만 의존하면 위험합니다

현재 `MEDASD.train()`에서 target 여부를 다음처럼 판단합니다.

```python
is_target = np.asarray(["target" in basename for basename in basenames], dtype=bool)
```

그리고 test에서도 domain을 `"target" in basename`으로 판단합니다. 

개발 데이터에서는 파일명에 source/target이 들어가면 괜찮을 수 있습니다. 하지만 evaluation dataset에서는 domain 정보가 제공되지 않습니다. 물론 evaluation에서는 metric 계산을 하지 않으므로 큰 문제는 아닐 수 있지만, train 쪽에서도 파일명 포맷이 바뀌면 깨질 수 있습니다.

이미 loader가 `condition`을 반환하고 있습니다. `__getitem__()`에서 `(waveform, y_true, condition, basename, index)`를 반환하므로, 가능하면 source/target 판별은 filename string보다 condition 또는 attribute parser를 쓰는 것이 더 안전합니다. 

최소 수정으로는 우선 다음처럼 basename fallback은 유지하되, 나중에 condition 기반으로 바꾸는 것이 좋습니다.

```python
# 임시
is_target = np.asarray(["target" in basename.lower() for basename in basenames], dtype=bool)
```

현재는 대소문자나 파일명 규칙에 취약합니다.

---

## 5. 반드시 고쳐야 할 부분 4: test loop가 batch size 1을 가정합니다

`test()` 안에서 다음 코드가 있습니다.

```python
basename = batch[3][0]
z = self.model(data).detach().cpu()
score = self.score_embeddings(z, stats).item()
y_true.append(batch[1][0].item())
```

이 구조는 test batch size가 1일 때만 안전합니다. batch size가 2 이상이면 `.item()`에서 에러가 나거나 첫 번째 basename만 저장하게 됩니다. 

안전하게 하려면 batch 전체를 처리해야 합니다.

```python
z = self.model(data).detach().cpu()
scores = self.score_embeddings(z, stats).cpu().numpy()

for bname, score, label in zip(batch[3], scores, batch[1].cpu().numpy()):
    y_pred.append(float(score))
    y_true.append(int(label))
    anomaly_score_list.append([bname, float(score)])
    decision_result_list.append([bname, 1 if score > decision_threshold else 0])
```

이 수정은 꼭 하시는 것이 좋습니다.

---

## 6. 확인이 필요한 부분: `self.valid_loader` 존재 여부

`train()`에서 다음 코드가 있습니다.

```python
if len(self.valid_loader) > 0:
```

BaseModel에서 항상 `valid_loader`가 정의된다면 괜찮습니다. 하지만 baseline 구조에 따라 valid loader가 없을 수 있으면 `AttributeError`가 날 수 있습니다. 

안전하게는 다음처럼 바꾸는 것이 좋습니다.

```python
if hasattr(self, "valid_loader") and self.valid_loader is not None and len(self.valid_loader) > 0:
```

---

## 7. `FixedEnvelope`에서 dtype/device는 거의 괜찮지만 minor 수정 권장

현재 `_analytic_signal()`에서 multiplier dtype을 `x.dtype`으로 만들고 complex spectrum에 곱합니다. PyTorch가 promotion을 해주기 때문에 보통 동작합니다. 그래도 명확하게 하려면 multiplier를 spectrum dtype으로 맞추는 편이 좋습니다.

```python
multiplier = torch.zeros(length, device=x.device, dtype=spectrum.dtype)
```

다만 이 경우 값은 complex가 됩니다. 더 깔끔한 방식은 현재처럼 float multiplier를 쓰고 promotion에 맡겨도 됩니다. 필수 수정은 아닙니다.

---

## 8. `diff = near - far`는 정규화 전략을 다시 생각해야 합니다

현재 waveform loader에서 전체 2채널 peak 기준으로 normalize합니다.

```python
peak = np.max(np.abs(wav))
if peak > 0:
    wav = wav / peak
```

이 방식은 두 채널의 상대 amplitude 차이를 보존합니다. DCASE 2026에서는 near/far의 SNR 차이가 중요하므로, **채널별 normalize가 아니라 전체 normalize를 한 것은 적절합니다.** 

다만 `diff = near - far`는 마이크 거리/위상 차이 때문에 단순 subtraction이 항상 좋은 것은 아닙니다. 첫 실험에서는 괜찮지만, 다음 ablation도 추천합니다.

```text
A: near only
B: near + far
C: near + far + diff
D: near + far + abs(near - far)
E: near + far + near/far log-mel difference
```

초기에는 지금처럼 `near/far/diff`가 맞습니다.

---

## 9. 지금 코드는 Fixed MED-ASD이지 LCMED는 아닙니다

현재 `network.py`는 `FixedEnvelope`만 있습니다. 즉, 이것은 **기존 MED 모듈을 이용한 방법**입니다. 아직 learnable cutoff인 LCMED는 아닙니다. 

LCMED로 가려면 다음 클래스를 추가해야 합니다.

```python
class LearnableSoftEnvelope(nn.Module):
    ...
```

그리고 `DiscriminatorEForASD`가 envelope module을 주입받도록 바꾸는 것이 좋습니다.

```python
class DiscriminatorEForASD(nn.Module):
    def __init__(self, envelope):
        self.envelope = envelope
```

그 다음:

```python
Fixed MED:
    FixedEnvelope(max_freq=300)

LCMED:
    LearnableSoftEnvelope(init_cutoff=300)
```

처럼 비교하면 됩니다.

---

# 최종 판단

현재 수정 방향은 **맞습니다.**
다만 지금 상태는 “실험을 돌릴 수 있는 초안”에 가깝고, 안정적으로 결과를 얻으려면 아래 4개는 꼭 수정하시는 것을 권장드립니다.

1. `pool_fmaps()`를 channel-wise pooling으로 바꾸거나, 현재 small pooling과 channel-wise pooling을 ablation으로 분리하십시오.
2. channel-wise pooling을 쓸 경우 PCA 또는 diagonal Mahalanobis를 추가하십시오.
3. `test()`가 batch size 1에 의존하지 않도록 수정하십시오.
4. source/target 판별을 `"target" in basename`에만 의존하지 않도록 개선하십시오.

가장 안전한 다음 단계는 다음 순서입니다.

```text
1. 현재 코드 그대로 small pooling으로 smoke test
2. batch size 1로 train/test 전체 pipeline 동작 확인
3. batch test loop 수정
4. channel-wise pooling + PCA/diagonal Mahalanobis 추가
5. near only / near+far / near+far+diff ablation
6. LCMED 추가
```

즉, **지금 형태로 MED-ASD 첫 버전은 시작해도 됩니다.**
하지만 성능을 기대하려면 pooling과 scoring 안정화가 핵심입니다.
