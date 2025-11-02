import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ==== configs ====
FEATURES = ["Temperature", "RH", "Ws", "Rain",
            "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
DEFAULT_LOG_DIR = "logs_dl"

# ==== tiện ích ====


@st.cache_resource
def load_artifacts(run_dir: Path):
    model_path = run_dir / "fire_model.pkl"
    scaler_path = run_dir / "fire_scaler.pkl"
    meta_path = run_dir / "train_meta.json"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Không tìm thấy fire_model.pkl hoặc fire_scaler.pkl trong thư mục đã chọn.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            try:
                meta = json.load(f)
            except Exception:
                meta = {}
    return model, scaler, meta


def predict_single(model, scaler, feat_vector, threshold=0.5):
    arr = np.asarray(feat_vector, dtype=float).reshape(1, -1)
    arr_s = scaler.transform(arr)
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(arr_s)[:, 1][0])
    else:
        scores = model.decision_function(arr_s).reshape(-1, 1)
        from sklearn.preprocessing import MinMaxScaler
        prob = float(MinMaxScaler().fit_transform(scores)[0, 0])
    label = "fire" if prob >= threshold else "not fire"
    return prob, label


def predict_batch(model, scaler, df_features: pd.DataFrame, threshold=0.5):
    X = scaler.transform(df_features.values)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X).reshape(-1, 1)
        from sklearn.preprocessing import MinMaxScaler
        prob = MinMaxScaler().fit_transform(scores).ravel()
    pred = (prob >= threshold).astype(int)
    label = np.where(pred == 1, "fire", "not fire")
    out = df_features.copy()
    out["prob_fire"] = prob
    out["pred"] = label
    return out


def list_available_runs(base_dir: Path):
    base_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for p in sorted(base_dir.iterdir()):
        if p.is_dir() and (p / "fire_model.pkl").exists() and (p / "fire_scaler.pkl").exists():
            runs.append(p)
    return runs


# ==== UI ====
st.set_page_config(page_title="Forest Fire Prediction", layout="centered")

st.title("Forest Fire Prediction — Inference App")
st.caption("Nhập 10 feature khí tượng / FWI để dự đoán khả năng cháy rừng.")

# Sidebar
with st.sidebar:
    st.header("Cấu hình")
    base_log_dir = st.text_input(
        "Thư mục chứa các run:", value=DEFAULT_LOG_DIR)
    run_paths = list_available_runs(Path(base_log_dir))
    if not run_paths:
        st.warning("Chưa phát hiện model nào trong thư mục logs/.")
        st.stop()
    run_labels = [p.name for p in run_paths]
    run_idx = st.selectbox("Chọn model", list(
        range(len(run_labels))), format_func=lambda i: run_labels[i])
    run_dir = run_paths[run_idx]

    try:
        model, scaler, meta = load_artifacts(run_dir)
        st.success(f"Đã load: {run_dir.name}")
    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        st.stop()

    default_th = float(meta.get("threshold", 0.5)
                       ) if isinstance(meta, dict) else 0.5
    threshold = st.slider("Ngưỡng (threshold)", 0.0, 1.0, default_th, 0.01)
    st.caption("Giảm threshold để ưu tiên recall (bắt cháy nhiều hơn).")

# ==== Single Prediction ====
st.subheader("Dự đoán từng điểm (Single Prediction)")
cols1 = st.columns(2)
with cols1[0]:
    Temperature = st.number_input("Temperature (°C)", value=30.0)
    RH = st.number_input("RH (%)", value=40.0)
    Ws = st.number_input("Ws (km/h)", value=10.0)
    Rain = st.number_input("Rain (mm)", value=0.0)
    FFMC = st.number_input("FFMC", value=85.0)
with cols1[1]:
    DMC = st.number_input("DMC", value=10.0)
    DC = st.number_input("DC", value=100.0)
    ISI = st.number_input("ISI", value=3.0)
    BUI = st.number_input("BUI", value=20.0)
    FWI = st.number_input("FWI", value=5.0)

if st.button("Dự đoán 1 điểm", use_container_width=True):
    vec = [Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, FWI]
    prob, label = predict_single(model, scaler, vec, threshold)
    st.success(f"**{label.upper()}**  |  Xác suất cháy = **{prob:.3f}**")
    st.progress(min(1.0, prob))

# ==== Batch Prediction ====
st.markdown("---")
st.subheader("Dự đoán từ CSV (Batch Prediction)")
st.caption(f"File CSV cần chứa tối thiểu các cột: {FEATURES}")
up = st.file_uploader("Tải CSV", type=["csv"])


def read_csv_flexible(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột cần thiết: {missing}")
    return df.copy()


if up is not None:
    try:
        df_in = read_csv_flexible(up)
        st.info(
            f"Đã đọc {len(df_in):,} dòng. Có các cột: {list(df_in.columns)}")

        if st.button("Dự đoán batch", use_container_width=True):
            df_pred = predict_batch(model, scaler, df_in[FEATURES], threshold)
            result = pd.concat(
                [df_in.reset_index(drop=True), df_pred[["prob_fire", "pred"]]], axis=1)
            st.success("Hoàn tất dự đoán.")
            st.dataframe(result.head(50))

            # Nếu có cột nhãn thật
            truth_cols = [c for c in df_in.columns if c.strip().lower(
            ) in ["classes", "class", "label", "fire", "target"]]
            if truth_cols:
                truth_col = truth_cols[0]
                y_true = df_in[truth_col].map(lambda x: 1 if str(x).strip().lower() in [
                                              "fire", "1", "true", "yes"] else 0)
                y_pred = (result["pred"] == "fire").astype(int)
                acc = (y_true == y_pred).mean()
                n_correct = int((y_true == y_pred).sum())
                n_total = len(y_true)
                st.markdown("### Đánh giá mô hình trên dữ liệu có nhãn")
                st.write(
                    f"**Độ chính xác:** {acc*100:.2f}%  ({n_correct}/{n_total})")

                cm = confusion_matrix(y_true, y_pred)
                st.write("**Confusion Matrix:**")
                st.dataframe(pd.DataFrame(cm, index=["Thực tế: Not Fire", "Thực tế: Fire"], columns=[
                             "Dự đoán: Not Fire", "Dự đoán: Fire"]))

                # Biểu đồ đúng/sai
                fig, ax = plt.subplots()
                ax.bar(["Đúng", "Sai"], [n_correct, n_total -
                       n_correct], color=["green", "red"])
                ax.set_title("Tỷ lệ đúng / sai")
                ax.set_ylabel("Số mẫu")
                st.pyplot(fig)

            else:
                st.info("Không tìm thấy cột nhãn thực tế (Classes / class / label).")

            csv_bytes = result.to_csv(index=False).encode("utf-8")
            st.download_button("Tải kết quả CSV", csv_bytes,
                               file_name=f"pred_{run_dir.name}.csv", mime="text/csv")

    except Exception as e:
        st.error(f"CSV không hợp lệ: {e}")

# ==== Thông tin chung ====
with st.expander("Thông tin run / meta"):
    st.json(meta if meta else {
            "info": "Không có train_meta.json hoặc không chứa thông tin thêm."})

st.caption("© Forest Fire Prediction • Streamlit App")
