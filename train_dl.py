# -*- coding: utf-8 -*-
import os, json, time
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
import joblib

FEATURES = ["Temperature","RH","Ws","Rain","FFMC","DMC","DC","ISI","BUI","FWI"]
TARGET = "Classes"

def load_df(csv_path):
    df = pd.read_csv(csv_path)
    df[TARGET] = df[TARGET].map(lambda x: 1 if str(x).strip().lower()=="fire" else 0)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=FEATURES+[TARGET])

def split_df(df, test_size=0.2, val_size=0.2, random_state=42):
    tr_df, te_df = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=random_state)
    val_ratio = val_size/(1-test_size)
    tr_df, val_df = train_test_split(tr_df, test_size=val_ratio, stratify=tr_df[TARGET], random_state=random_state)
    return tr_df.reset_index(drop=True), val_df.reset_index(drop=True), te_df.reset_index(drop=True)

def balance_near_equal(df, delta=100, random_state=42):
    fire_df = df[df[TARGET]==1]
    notfire_df = df[df[TARGET]==0]
    n_fire = len(fire_df)
    n_not_target = min(len(notfire_df), n_fire + max(0,int(delta)))
    notfire_keep = notfire_df.sample(n=n_not_target, random_state=random_state, replace=False)
    return pd.concat([fire_df, notfire_keep], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

class TabDataset(Dataset):
    def __init__(self,X,y): self.X=torch.tensor(X,dtype=torch.float32); self.y=torch.tensor(y,dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.X[i],self.y[i]

class MLP(nn.Module):
    def __init__(self,in_dim,hidden=256,dropout=0.2):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim,hidden),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(hidden,hidden//2),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(hidden//2,2))
    def forward(self,x): return self.net(x)

class LSTMHead(nn.Module):
    def __init__(self, feat_dim, hidden=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))
    def forward(self,x):
        # nếu input 2D (B,F) -> thêm time-step = 1: (B,1,F)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h,_ = self.lstm(x)              # (B,T,H)
        return self.head(h[:,-1,:])     # (B,2)

# class TCNBlock(nn.Module):
#     def __init__(self, in_c, out_c, k=3, d=1):
#         super().__init__()
#         pad=(k-1)*d
#         self.net=nn.Sequential(
#             nn.Conv1d(in_c,out_c,k,padding=pad,dilation=d),
#             nn.ReLU(),
#             nn.Conv1d(out_c,out_c,k,padding=pad,dilation=d),
#             nn.ReLU()
#         )
#         self.proj=nn.Conv1d(in_c,out_c,1) if in_c!=out_c else nn.Identity()
#     def forward(self,x):
#         y=self.net(x)
#         return y + self.proj(x)

# class TCNHead(nn.Module):
#     def __init__(self, feat_dim, hidden=128):
#         super().__init__()
#         self.block1=TCNBlock(feat_dim,hidden,d=1)
#         self.block2=TCNBlock(hidden,hidden,d=2)
#         self.pool=nn.AdaptiveAvgPool1d(1)
#         self.fc=nn.Linear(hidden,2)
#     def forward(self,x):
#         # nếu input 2D (B,F) -> (B,1,F) rồi chuyển thành (B,F,T)
#         if x.dim() == 2:
#             x = x.unsqueeze(1)
#         x = x.transpose(1,2)            # (B,F,T)
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.pool(x).squeeze(-1)    # (B,H)
#         return self.fc(x)               # (B,2)

def build_scaler(name): return StandardScaler() if name=="standard" else MinMaxScaler()

def train_epoch(model,loader,crit,opt,device):
    model.train();loss_sum=0
    for xb,yb in loader:
        xb,yb=xb.to(device),yb.to(device)
        opt.zero_grad();logits=model(xb);loss=crit(logits,yb);loss.backward();opt.step()
        loss_sum+=loss.item()*len(yb)
    return loss_sum/len(loader.dataset)

@torch.no_grad()
def eval_epoch(model,loader,crit,device):
    model.eval();loss_sum=0;probs=[];ys=[]
    for xb,yb in loader:
        xb,yb=xb.to(device),yb.to(device)
        logits=model(xb);loss=crit(logits,yb)
        loss_sum+=loss.item()*len(yb)
        p=torch.softmax(logits,dim=1)[:,1].cpu().numpy()
        probs.append(p);ys.append(yb.cpu().numpy())
    return loss_sum/len(loader.dataset),np.concatenate(probs),np.concatenate(ys)

def tune_threshold_min_precision(y_true, prob, min_precision=0.3):
    p,r,t=precision_recall_curve(y_true,prob);p,r=p[:-1],r[:-1]
    if len(t)==0:return 0.5,float(p.mean()),float(r.mean())
    mask=p>=min_precision
    if not np.any(mask):
        f1=2*p*r/(p+r+1e-12);i=int(np.argmax(f1));return float(t[i]),float(p[i]),float(r[i])
    idxs=np.where(mask)[0];i=int(idxs[np.argmax(r[idxs])]);return float(t[i]),float(p[i]),float(r[i])

def plot_curves(y_true,prob,y_pred,out_dir,title):
    out_dir.mkdir(parents=True,exist_ok=True)
    prec,rec,_=precision_recall_curve(y_true,prob)
    plt.figure();plt.plot(rec,prec);plt.xlabel("Recall");plt.ylabel("Precision");plt.title(f"PR: {title}");plt.tight_layout();plt.savefig(out_dir/"pr_curve.png");plt.close()
    fpr,tpr,_=roc_curve(y_true,prob);roc_auc=auc(fpr,tpr)
    plt.figure();plt.plot(fpr,tpr,label=f"AUC={roc_auc:.3f}");plt.plot([0,1],[0,1],'--');plt.legend();plt.xlabel("FPR");plt.ylabel("TPR");plt.title(f"ROC: {title}");plt.tight_layout();plt.savefig(out_dir/"roc_curve.png");plt.close()
    cm=confusion_matrix(y_true,y_pred);plt.figure();im=plt.imshow(cm,cmap="Blues");plt.colorbar(im)
    plt.xticks([0,1],["not fire","fire"]);plt.yticks([0,1],["not fire","fire"])
    for i in range(2):
        for j in range(2):plt.text(j,i,f"{cm[i,j]:,}",ha="center",va="center")
    plt.title(f"CM: {title}");plt.tight_layout();plt.savefig(out_dir/"confusion_matrix.png");plt.close()

def to_py(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

def run_one(cfg,model_name,scaler_name):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name=f"{model_name}_{scaler_name}"
    run_dir=Path(cfg["log_dir"])/run_name;run_dir.mkdir(parents=True,exist_ok=True)
    log_path=run_dir/f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    def log(s):print(s);open(log_path,"a").write(s+"\n")

    df=load_df(cfg["csv_path"])
    tr,val,te=split_df(df,cfg["test_size"],cfg["val_size"],cfg["random_state"])
    tr=balance_near_equal(tr,cfg["balance_delta"],cfg["random_state"])

    scaler=build_scaler(scaler_name)
    Xtr=scaler.fit_transform(tr[FEATURES]).astype(np.float32)
    Xval=scaler.transform(val[FEATURES]).astype(np.float32)
    Xte=scaler.transform(te[FEATURES]).astype(np.float32)
    ytr,yval,yte=tr[TARGET].values,val[TARGET].values,te[TARGET].values
    joblib.dump(scaler,run_dir/"scaler.pkl")

    train_loader=DataLoader(TabDataset(Xtr,ytr),batch_size=cfg["batch_size"],shuffle=True)
    val_loader=DataLoader(TabDataset(Xval,yval),batch_size=cfg["batch_size"])
    test_loader=DataLoader(TabDataset(Xte,yte),batch_size=cfg["batch_size"])

    if model_name=="lstm":
        model=LSTMHead(len(FEATURES)).to(device)
    else:
        model=MLP(len(FEATURES),cfg["hidden"],cfg["dropout"]).to(device)

    pos,neg=(ytr==1).sum(),(ytr==0).sum()
    w=torch.tensor([1.0,neg/max(1,pos)],dtype=torch.float32,device=device)
    crit=nn.CrossEntropyLoss(weight=w)
    opt=torch.optim.Adam(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])

    best_val=float("inf");best_state=None;no_imp=0;t0=time.time()
    for ep in range(1,cfg["epochs"]+1):
        trl=train_epoch(model,train_loader,crit,opt,device)
        vall,_,_=eval_epoch(model,val_loader,crit,device)
        if vall<best_val-1e-4:best_val=vall;best_state={k:v.cpu().clone() for k,v in model.state_dict().items()};no_imp=0
        else:no_imp+=1
        if ep%cfg["log_every"]==0 or ep==1:log(f"{run_name} epoch {ep:03d} tr={trl:.4f} val={vall:.4f}")
        if no_imp>=cfg["early_stop"]:log("early stop");break
    model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
    torch.save(model.state_dict(),run_dir/"model.pt")

    _,val_prob,y_val=eval_epoch(model,val_loader,crit,device)
    th,p_at,r_at=tune_threshold_min_precision(y_val,val_prob,cfg["min_precision"])
    _,te_prob,y_te=eval_epoch(model,test_loader,crit,device)
    y_pred=(te_prob>=th).astype(int)
    rep=classification_report(y_te,y_pred,target_names=["not fire","fire"],digits=3)
    cm=confusion_matrix(y_te,y_pred)
    log(rep);log(str(cm))
    plot_curves(y_te,te_prob,y_pred,run_dir,run_name)

    meta=dict(cfg)
    meta.update(dict(
        model=model_name, scaler=scaler_name,
        threshold=float(th), val_precision=float(p_at), val_recall=float(r_at),
        train_time_sec=float(round(time.time()-t0,2))
    ))
    meta_py={k:to_py(v) for k,v in meta.items()}
    with open(run_dir/"meta.json","w",encoding="utf-8") as f:
        json.dump(meta_py,f,indent=2,ensure_ascii=False)

def main():
    BASE_CONFIG=dict(
        csv_path="data/data/clean/clean_2000_2024_timelines.csv",
        test_size=0.2,
        val_size=0.2,
        balance_delta=0,
        random_state=42,
        batch_size=2048,
        epochs=50,
        early_stop=5,
        log_every=1,
        hidden=256,
        dropout=0.2,
        lr=1e-3,
        weight_decay=1e-5,
        min_precision=0.1,
        log_dir="logs_dl"
    )
    MODELS=["mlp","lstm"]
    SCALERS=["standard","minmax"]
    for m in MODELS:
        for s in SCALERS:
            run_one(BASE_CONFIG,m,s)

if __name__=="__main__":
    main()
