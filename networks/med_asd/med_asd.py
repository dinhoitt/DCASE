import csv
import os
import sys

import numpy as np
import scipy
import torch
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from networks.base_model import BaseModel
from networks.med_asd.network import MEDFeatureExtractor
from tools.plot_anm_score import AnmScoreFigData


class MEDASD(BaseModel):
    """Fixed-cutoff MED-ASD with source/target Mahalanobis scoring."""

    def __init__(self, args, train, test):
        super().__init__(args=args, train=train, test=test)
        self.stats_path = self.model_dir / (
            f"stats_{self.args.model}_{self.args.dataset}{self.model_name_suffix}"
            f"{self.eval_suffix}_seed{self.args.seed}.pth"
        )
        self.score_distr_file_path = self.model_dir / (
            f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}"
            f"{self.eval_suffix}_seed{self.args.seed}_mahala.pickle"
        )

    def init_model(self):
        cutoffs = tuple(float(value) for value in self.args.med_cutoffs)
        return MEDFeatureExtractor(cutoff_list=cutoffs, sample_rate=self.args.med_sample_rate)

    def get_log_header(self):
        return "loss,source_count,target_count,embedding_dim,pca_dim"

    def train(self, epoch):
        if epoch > 1:
            return

        print("\n============== FIT MED-ASD MAHALANOBIS ==============")
        self.model.eval()
        embeddings, basenames = self.extract_embeddings(self.train_loader)
        is_target = np.asarray(["target" in basename.lower() for basename in basenames], dtype=bool)
        is_source = np.logical_not(is_target)

        transform = self.fit_embedding_transform(embeddings)
        embeddings_pca = self.apply_embedding_transform(embeddings, transform)
        source_embeddings = embeddings_pca[is_source]
        target_embeddings = embeddings_pca[is_target]
        source_stats = self.fit_gaussian(source_embeddings)
        target_stats = self.fit_gaussian(target_embeddings if len(target_embeddings) > 1 else source_embeddings)

        stats = {
            "source": source_stats,
            "target": target_stats,
            "cutoffs": self.args.med_cutoffs,
            "sample_rate": self.args.med_sample_rate,
            "audio_samples": self.args.med_audio_samples,
            "embedding_transform": transform,
        }
        torch.save(stats, self.stats_path)
        torch.save(self.model.state_dict(), self.model_path)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "stats": stats,
                "loss": torch.tensor(0.0),
            },
            self.checkpoint_path,
        )

        train_scores = self.score_embeddings(embeddings_pca, stats).cpu().numpy().tolist()
        if hasattr(self, "valid_loader") and self.valid_loader is not None and len(self.valid_loader) > 0:
            valid_embeddings, _ = self.extract_embeddings(self.valid_loader)
            valid_embeddings_pca = self.apply_embedding_transform(valid_embeddings, transform)
            train_scores.extend(self.score_embeddings(valid_embeddings_pca, stats).cpu().numpy().tolist())
        self.fit_anomaly_score_distribution(
            y_pred=train_scores,
            score_distr_file_path=self.score_distr_file_path,
        )

        with open(self.log_path, "a") as log:
            np.savetxt(
                log,
                [
                    "{0},{1},{2},{3},{4}".format(
                        0.0,
                        int(is_source.sum()),
                        int(is_target.sum()),
                        int(embeddings.shape[1]),
                        int(embeddings_pca.shape[1]),
                    )
                ],
                fmt="%s",
            )

    def extract_embeddings(self, loader):
        zs = []
        basenames = []
        with torch.no_grad():
            for batch in tqdm(loader):
                data = batch[0].to(self.device).float()
                z = self.model(data).detach().cpu()
                zs.append(z)
                basenames.extend(list(batch[3]))
        return torch.cat(zs, dim=0), basenames

    def fit_embedding_transform(self, embeddings):
        embeddings_np = embeddings.numpy()
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_np)
        max_components = min(self.args.med_pca_dim, embeddings_scaled.shape[0] - 1, embeddings_scaled.shape[1])
        if max_components < 1:
            raise ValueError("PCA needs at least two training embeddings for MED-ASD.")

        pca = PCA(n_components=max_components, svd_solver="auto", random_state=self.args.seed)
        pca.fit(embeddings_scaled)
        return {
            "scaler_mean": torch.from_numpy(scaler.mean_.astype(np.float32)),
            "scaler_scale": torch.from_numpy(scaler.scale_.astype(np.float32)),
            "pca_mean": torch.from_numpy(pca.mean_.astype(np.float32)),
            "pca_components": torch.from_numpy(pca.components_.astype(np.float32)),
            "pca_explained_variance_ratio": torch.from_numpy(pca.explained_variance_ratio_.astype(np.float32)),
        }

    @staticmethod
    def apply_embedding_transform(embeddings, transform):
        mean = transform["scaler_mean"]
        scale = torch.clamp(transform["scaler_scale"], min=1e-12)
        scaled = (embeddings - mean) / scale
        centered = scaled - transform["pca_mean"]
        return centered.matmul(transform["pca_components"].t())

    def fit_gaussian(self, embeddings):
        if len(embeddings) == 0:
            raise ValueError("Cannot fit MED-ASD Gaussian stats with no embeddings.")
        mu = embeddings.mean(dim=0)
        centered = embeddings - mu
        denom = max(len(embeddings) - 1, 1)
        cov = centered.t().matmul(centered) / denom
        eps = self.args.med_cov_eps
        cov = cov + eps * torch.eye(cov.shape[0], dtype=cov.dtype)
        inv_cov = torch.linalg.pinv(cov)
        return {"mu": mu, "inv_cov": inv_cov}

    @staticmethod
    def mahalanobis(embeddings, stats):
        mu = stats["mu"]
        inv_cov = stats["inv_cov"]
        delta = embeddings - mu
        return torch.sum(delta.matmul(inv_cov) * delta, dim=1)

    def score_embeddings(self, embeddings, stats):
        source_score = self.mahalanobis(embeddings, stats["source"])
        target_score = self.mahalanobis(embeddings, stats["target"])
        return torch.minimum(source_score, target_score)

    def test(self):
        anm_score_figdata = AnmScoreFigData()
        mode = self.data.mode
        csv_lines = []

        print("============== MED-ASD MODEL LOAD ==============")
        if not os.path.exists(self.model_path):
            print(f"model not found -> {self.model_path}")
        else:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        stats = torch.load(self.stats_path, map_location="cpu")
        self.model.eval()
        decision_threshold = self.calc_decision_threshold(score_distr_file_path=self.score_distr_file_path)

        dir_name = "test"
        if mode:
            performance_over_all = []
            performance = []

        for idx, test_loader_tmp in enumerate(self.test_loader):
            section_name = f"section_{self.data.section_id_list[idx]}"
            result_dir = self.result_dir if self.args.dev else self.eval_data_result_dir
            anomaly_score_csv = result_dir / (
                f"anomaly_score_{self.args.dataset}_{section_name}_{dir_name}_seed{self.args.seed}"
                f"{self.model_name_suffix}{self.eval_suffix}.csv"
            )
            decision_result_csv = result_dir / (
                f"decision_result_{self.args.dataset}_{section_name}_{dir_name}_seed{self.args.seed}"
                f"{self.model_name_suffix}{self.eval_suffix}.csv"
            )

            y_pred = []
            y_true = []
            domain_list = [] if mode else None
            anomaly_score_list = []
            decision_result_list = []

            print("\n============== BEGIN MED-ASD TEST FOR A SECTION ==============")
            with torch.no_grad():
                for batch in tqdm(test_loader_tmp):
                    data = batch[0].to(self.device).float()
                    z = self.model(data).detach().cpu()
                    z = self.apply_embedding_transform(z, stats["embedding_transform"])
                    scores = self.score_embeddings(z, stats).cpu().numpy()
                    labels = batch[1].cpu().numpy()
                    for basename, score, label in zip(batch[3], scores, labels):
                        score = float(score)
                        y_pred.append(score)
                        y_true.append(int(label))
                        anomaly_score_list.append([basename, score])
                        decision_result_list.append([basename, 1 if score > decision_threshold else 0])
                        if mode:
                            domain_list.append("target" if "target" in basename.lower() else "source")

            save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            print(f"anomaly score result ->  {anomaly_score_csv}")
            save_csv(save_file_path=decision_result_csv, save_data=decision_result_list)
            print(f"decision result ->  {decision_result_csv}")

            if mode:
                self.append_metrics(
                    csv_lines=csv_lines,
                    performance=performance,
                    performance_over_all=performance_over_all,
                    anm_score_figdata=anm_score_figdata,
                    section_name=section_name,
                    y_true=y_true,
                    y_pred=y_pred,
                    domain_list=domain_list,
                    decision_threshold=decision_threshold,
                )
            print("\n============ END OF MED-ASD TEST FOR A SECTION ============")

        if not mode:
            return

        amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
        csv_lines.append(["arithmetic mean"] + list(amean_performance))
        hmean_performance = scipy.stats.hmean(
            np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon),
            axis=0,
        )
        csv_lines.append(["harmonic mean"] + list(hmean_performance))
        csv_lines.append([])
        anm_score_figdata.show_fig(
            title=self.args.model + "_" + self.args.dataset + self.model_name_suffix + self.eval_suffix + "_anm_score",
            export_dir=result_dir,
        )
        result_path = result_dir / (
            f"result_{self.args.dataset}_{dir_name}_seed{self.args.seed}"
            f"{self.model_name_suffix}{self.eval_suffix}_roc.csv"
        )
        print(f"results -> {result_path}")
        save_csv(save_file_path=result_path, save_data=csv_lines)

    def append_metrics(
        self,
        csv_lines,
        performance,
        performance_over_all,
        anm_score_figdata,
        section_name,
        y_true,
        y_pred,
        domain_list,
        decision_threshold,
    ):
        y_true_s_auc = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "source" or y_true[idx] == 1]
        y_pred_s_auc = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "source" or y_true[idx] == 1]
        y_true_t_auc = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "target" or y_true[idx] == 1]
        y_pred_t_auc = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "target" or y_true[idx] == 1]
        y_true_s = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "source"]
        y_pred_s = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "source"]
        y_true_t = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "target"]
        y_pred_t = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "target"]

        auc_s = metrics.roc_auc_score(y_true_s_auc, y_pred_s_auc)
        p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=self.args.max_fpr)
        p_auc_s = metrics.roc_auc_score(y_true_s, y_pred_s, max_fpr=self.args.max_fpr)
        tn_s, fp_s, fn_s, tp_s = metrics.confusion_matrix(
            y_true_s, [1 if x > decision_threshold else 0 for x in y_pred_s]
        ).ravel()
        prec_s = tp_s / np.maximum(tp_s + fp_s, sys.float_info.epsilon)
        recall_s = tp_s / np.maximum(tp_s + fn_s, sys.float_info.epsilon)
        f1_s = 2.0 * prec_s * recall_s / np.maximum(prec_s + recall_s, sys.float_info.epsilon)

        anm_score_figdata.append_figdata(
            anm_score_figdata.anm_score_to_figdata(
                scores=[[t, p] for t, p in zip(y_true_s, y_pred_s)],
                title=f"{section_name}_source_AUC{auc_s}",
            )
        )
        print(f"AUC (source) : {auc_s}")
        print(f"pAUC : {p_auc}")
        print(f"pAUC (source) : {p_auc_s}")

        if len(y_true_t) > 0:
            auc_t = metrics.roc_auc_score(y_true_t_auc, y_pred_t_auc)
            p_auc_t = metrics.roc_auc_score(y_true_t, y_pred_t, max_fpr=self.args.max_fpr)
            tn_t, fp_t, fn_t, tp_t = metrics.confusion_matrix(
                y_true_t, [1 if x > decision_threshold else 0 for x in y_pred_t]
            ).ravel()
            prec_t = tp_t / np.maximum(tp_t + fp_t, sys.float_info.epsilon)
            recall_t = tp_t / np.maximum(tp_t + fn_t, sys.float_info.epsilon)
            f1_t = 2.0 * prec_t * recall_t / np.maximum(prec_t + recall_t, sys.float_info.epsilon)
            if len(csv_lines) == 0:
                csv_lines.append(self.result_column_dict["source_target"])
            csv_lines.append(
                [section_name.split("_", 1)[1], auc_s, auc_t, p_auc, p_auc_s, p_auc_t, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t]
            )
            performance.append([auc_s, auc_t, p_auc, p_auc_s, p_auc_t, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])
            performance_over_all.append([auc_s, auc_t, p_auc, p_auc_s, p_auc_t, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])
            print(f"AUC (target) : {auc_t}")
            print(f"pAUC (target) : {p_auc_t}")
        else:
            if len(csv_lines) == 0:
                csv_lines.append(self.result_column_dict["single_domain"])
            csv_lines.append([section_name.split("_", 1)[1], auc_s, p_auc, prec_s, recall_s, f1_s])
            performance.append([auc_s, p_auc, prec_s, recall_s, f1_s])
            performance_over_all.append([auc_s, p_auc, prec_s, recall_s, f1_s])


def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(save_data)
