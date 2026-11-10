"""Figures, the Excel cut-off model and the Power BI exports."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from openpyxl.chart import LineChart, Reference  # noqa: E402
from openpyxl.comments import Comment  # noqa: E402
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

# Validated categorical slots 1-2 (blue, orange) and text/grid tokens.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#9c9a92"
MODEL_COLORS = {"Scorecard": BLUE, "Gradient boosting": ORANGE}


def _style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def figures(fig_dir: Path, *, y_te, pd_card, pd_gbm, iv, importance, cut_card, cut_gbm, views, summary):
    fig_dir.mkdir(parents=True, exist_ok=True)
    m = summary["models"]

    # ROC ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=1, ls="--")
    for name, p, key in (("Scorecard", pd_card, "scorecard"), ("Gradient boosting", pd_gbm, "gbm_calibrated")):
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=2,
                label=f"{name}  AUC {m[key]['auc']:.3f} · Gini {m[key]['gini']:.3f} · KS {m[key]['ks']:.3f}")
    _style(ax, "ROC on the held-out test set", "False positive rate", "True positive rate")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    _save(fig, fig_dir / "roc.png")

    # Calibration ----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    lim = 0
    for name, p in (("Scorecard", pd_card), ("Gradient boosting", pd_gbm)):
        d = pd.DataFrame({"y": y_te, "p": p})
        d["b"] = pd.qcut(d["p"].rank(method="first"), 10, labels=False)
        g = d.groupby("b").agg(p=("p", "mean"), y=("y", "mean"))
        lim = max(lim, g.max().max())
        ax.plot(g["p"], g["y"], color=MODEL_COLORS[name], lw=2, marker="o", ms=5, label=name)
    ax.plot([0, lim * 1.05], [0, lim * 1.05], color=NEUTRAL, lw=1, ls="--", label="Perfect calibration")
    _style(ax, "Calibration by predicted-PD decile", "Mean predicted PD", "Observed default rate")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    _save(fig, fig_dir / "calibration.png")

    # IV -------------------------------------------------------------------------------
    top = iv.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.barh(top["feature"], top["iv"], color=BLUE, height=0.65)
    for yv, v in enumerate(top["iv"]):
        ax.text(v + 0.003, yv, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
    _style(ax, "Information Value – top 15 characteristics", "Information Value")
    ax.grid(axis="y", visible=False)
    _save(fig, fig_dir / "iv_top15.png")

    # Permutation importance -------------------------------------------------------------
    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.barh(top["feature"], top["auc_drop"], xerr=top["std"], color=ORANGE, height=0.65,
            error_kw={"ecolor": INK2, "elinewidth": 0.8})
    _style(ax, "Gradient boosting – permutation importance", "Drop in test AUC when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    _save(fig, fig_dir / "gbm_importance.png")

    # Cut-off trade-off: two panels, one measure each (no dual axis) ---------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, t in (("Scorecard", cut_card), ("Gradient boosting", cut_gbm)):
        axes[0].plot(t["approval_rate"], t["bad_rate"], color=MODEL_COLORS[name], lw=2, label=name)
        axes[1].plot(t["approval_rate"], t["profit"] / 1e9, color=MODEL_COLORS[name], lw=2, label=name)
    best = cut_card.loc[cut_card["profit"].idxmax()]
    axes[1].scatter([best["approval_rate"]], [best["profit"] / 1e9], color=BLUE, s=60, zorder=5,
                    edgecolor="white", linewidth=2)
    axes[1].annotate(f"Scorecard optimum\n{best['approval_rate']:.0%} approved",
                     (best["approval_rate"], best["profit"] / 1e9), xytext=(0.55, 0.55), fontsize=8, color=INK,
                     arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    _style(axes[0], "Bad rate of approved book", "Approval rate", "Default rate")
    _style(axes[1], "Portfolio profit (test set)", "Approval rate", "Profit (billion, data currency)")
    for ax in axes:
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    _save(fig, fig_dir / "cutoff_tradeoff.png")

    # Segment default rates --------------------------------------------------------------
    seg = pd.concat([views["by_bureau_history"].assign(group="Bureau history"),
                     views["by_income_type"].assign(group="Income type")])
    seg = seg.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(seg["segment"], seg["default_rate"], color=BLUE, height=0.6)
    for yv, (v, n) in enumerate(zip(seg["default_rate"], seg["applications"], strict=True)):
        ax.text(v + 0.001, yv, f"{v:.1%}  (n={n:,})", va="center", fontsize=7.5, color=INK2)
    ax.axvline(summary["data"]["default_rate"], color=NEUTRAL, lw=1, ls="--")
    _style(ax, "Default rate by segment")
    ax.text(summary["data"]["default_rate"], len(seg) - 0.35, f" portfolio avg {summary['data']['default_rate']:.1%}",
            fontsize=7.5, color=INK2, va="bottom")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, seg["default_rate"].max() * 1.45)
    _save(fig, fig_dir / "segments.png")


# --------------------------------------------------------------------------------------
# Excel cut-off model with live formulas
# --------------------------------------------------------------------------------------
FONT = "Arial"
F_IN = Font(name=FONT, color="0000FF")
F_LINK = Font(name=FONT, color="008000")
F_TXT = Font(name=FONT)
F_BOLD = Font(name=FONT, bold=True)
F_HEAD = Font(name=FONT, bold=True, color="FFFFFF")
F_TITLE = Font(name=FONT, bold=True, size=14)
FILL_HEAD = PatternFill("solid", fgColor="1F3864")
FILL_KEY = PatternFill("solid", fgColor="FFFF00")
THIN = Border(bottom=Side(style="thin", color="BFBFBF"))
MONEY = '#,##0;(#,##0);"-"'
PCT = '0.0%;(0.0%);"-"'


def _cutoff_sheet(wb: Workbook, title: str, t: pd.DataFrame, model_label: str,
                  cutoff_label: str = "Score cut-off", cutoff_fmt: str = "0.0"):
    ws = wb.create_sheet(title)
    ws["A1"], ws["A1"].font = f"Cut-off strategy – {model_label} (test set, {int(t['approved'].max()):,} applications)", F_TITLE
    ws["A2"], ws["A2"].font = ("Black = formula, green = link to Inputs. Change LGD / margin on the Inputs sheet; "
                               "every column to the right of F recalculates."), Font(name=FONT, italic=True, color="595959")
    head = ["Approval rate", cutoff_label, "Approved (#)", "Approved defaults (#)",
            "Exposure approved", "Exposure of defaults", "Bad rate", "Expected loss",
            "Revenue", "Profit", "Loss rate", "Profit vs approve-all"]
    r0 = 4
    for j, h in enumerate(head, 1):
        c = ws.cell(r0, j, h)
        c.font, c.fill = F_HEAD, FILL_HEAD
        c.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
    ws.row_dimensions[r0].height = 32
    n = len(t)
    last = r0 + n
    for i, row in enumerate(t.itertuples(index=False), start=r0 + 1):
        vals = [row.approval_rate, row.score_cutoff, row.approved, row.approved_bads,
                row.exposure_approved, row.exposure_bad]
        for j, v in enumerate(vals, 1):
            ws.cell(i, j, float(v)).font = F_TXT
        ws.cell(i, 7, f"=IFERROR(D{i}/C{i},0)")
        ws.cell(i, 8, f"=Inputs!$C$5*F{i}")
        ws.cell(i, 9, f"=Inputs!$C$6*(E{i}-F{i})")
        ws.cell(i, 10, f"=I{i}-H{i}")
        ws.cell(i, 11, f"=IFERROR(H{i}/E{i},0)")
        ws.cell(i, 12, f"=J{i}-$J${last}")
        for j in range(7, 13):
            ws.cell(i, j).font = F_LINK if j in (8, 9) else F_TXT
        for j, fmt in ((1, "0%"), (2, cutoff_fmt), (3, "#,##0"), (4, "#,##0"), (5, MONEY), (6, MONEY),
                       (7, PCT), (8, MONEY), (9, MONEY), (10, MONEY), (11, PCT), (12, MONEY)):
            ws.cell(i, j).number_format = fmt
            ws.cell(i, j).border = THIN
    # Summary block driven by formulas
    s = last + 2
    rng = f"$J${r0 + 1}:$J${last}"
    items = [
        ("Profit-maximising approval rate", f"=INDEX($A${r0 + 1}:$A${last},MATCH(MAX({rng}),{rng},0))", "0%"),
        (f"{cutoff_label} at that rate", f"=INDEX($B${r0 + 1}:$B${last},MATCH(MAX({rng}),{rng},0))", cutoff_fmt),
        ("Bad rate of approved book", f"=INDEX($G${r0 + 1}:$G${last},MATCH(MAX({rng}),{rng},0))", PCT),
        ("Maximum profit", f"=MAX({rng})", MONEY),
        ("Profit if every applicant is approved", f"=$J${last}", MONEY),
        ("Uplift vs approve-all", f"=MAX({rng})-$J${last}", MONEY),
        ("Uplift vs approve-all (%)", f"=IFERROR((MAX({rng})-$J${last})/ABS($J${last}),0)", PCT),
    ]
    for k, (label, f, fmt) in enumerate(items):
        ws.cell(s + k, 1, label).font = F_BOLD
        c = ws.cell(s + k, 5, f)
        c.font, c.number_format = F_BOLD, fmt
    widths = [12, 11, 12, 13, 18, 18, 10, 17, 17, 17, 10, 18]
    for j, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = ws.cell(r0 + 1, 2)

    ch = LineChart()
    ch.title = "Profit by approval rate"
    ch.y_axis.title, ch.x_axis.title = "Profit", "Approval rate"
    ch.add_data(Reference(ws, min_col=10, min_row=r0, max_row=last), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=r0 + 1, max_row=last))
    ch.series[0].graphicalProperties.line.solidFill = BLUE.lstrip("#")
    ch.series[0].graphicalProperties.line.width = 25000
    ch.legend = None
    ch.height, ch.width = 8, 16
    ws.add_chart(ch, f"N{r0}")
    return ws


def excel_model(path: Path, *, cut_card, cut_gbm, summary, views):
    wb = Workbook()
    ws = wb.active
    ws.title = "Inputs"
    ws["A1"], ws["A1"].font = "Credit cut-off strategy model – Home Credit Default Risk", F_TITLE
    ws["A3"], ws["A3"].font = "Assumptions (edit the yellow cells)", F_BOLD
    a = summary["strategy"]["assumptions"]
    rows = [
        (5, "Loss given default (LGD)", a["lgd"],
         "Share of a defaulted loan's exposure that is never recovered. 60% is a working assumption for "
         "unsecured consumer credit; replace with the lender's own recovery data."),
        (6, "Net margin on repaid loans", a["net_margin"],
         "Lifetime interest + fees net of funding and servicing cost, as a share of exposure. Working "
         "assumption; replace with product P&L."),
    ]
    for r, label, v, note in rows:
        ws.cell(r, 2, label).font = F_TXT
        c = ws.cell(r, 3, v)
        c.font, c.fill, c.number_format = F_IN, FILL_KEY, "0.0%"
        c.comment = Comment(note, "model")
        ws.cell(r, 4, note).font = Font(name=FONT, italic=True, color="595959", size=9)
    ws["A8"], ws["A8"].font = "How the sheets work", F_BOLD
    notes = [
        "Applicants in the held-out test set are ranked best-first by score and approved down to each approval rate.",
        "Expected loss = LGD x exposure of approved loans that defaulted.  Revenue = margin x exposure of approved loans that repaid.",
        "Profit = Revenue - Expected loss.  The summary under each table finds the profit-maximising cut-off with INDEX/MATCH.",
        "Exposure is AMT_CREDIT in the dataset's own currency units (the dataset does not name the currency).",
        "Caveat: the data only contains loans that were granted, so rejected applicants are not observed (no reject inference).",
    ]
    for i, t in enumerate(notes, 9):
        ws.cell(i, 2, t).font = F_TXT
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 3, 30
    ws.column_dimensions["C"].width, ws.column_dimensions["D"].width = 10, 100

    _cutoff_sheet(wb, "Scorecard cut-offs", cut_card, "WoE logistic scorecard")
    _cutoff_sheet(wb, "GBM cut-offs", cut_gbm, "gradient boosting (calibrated PD)",
                  cutoff_label="Max PD approved", cutoff_fmt="0.0%")

    # Model comparison (static results from the pipeline)
    mc = wb.create_sheet("Model comparison")
    mc["A1"], mc["A1"].font = "Model performance on the held-out test set", F_TITLE
    hdr = ["Metric", "Scorecard", "Gradient boosting (calibrated)"]
    for j, h in enumerate(hdr, 1):
        c = mc.cell(3, j, h)
        c.font, c.fill = F_HEAD, FILL_HEAD
    m = summary["models"]
    for i, (k, lab, fmt) in enumerate((("auc", "AUC", "0.000"), ("gini", "Gini", "0.000"), ("ks", "KS", "0.000"),
                                       ("brier", "Brier score", "0.0000"), ("ece", "Calibration error (ECE)", "0.00%"),
                                       ("mean_pd", "Mean predicted PD", "0.00%"),
                                       ("observed_default_rate", "Observed default rate", "0.00%")), 4):
        mc.cell(i, 1, lab).font = F_TXT
        for j, key in ((2, "scorecard"), (3, "gbm_calibrated")):
            c = mc.cell(i, j, m[key][k])
            c.font, c.number_format = F_TXT, fmt
    mc.cell(12, 1, "Source: pipeline output reports/summary.json (static values).").font = Font(name=FONT, italic=True, size=9)
    for col, w in (("A", 28), ("B", 14), ("C", 30)):
        mc.column_dimensions[col].width = w

    # Segments
    sg = wb.create_sheet("Segments")
    sg["A1"], sg["A1"].font = "Default rate by segment (full portfolio)", F_TITLE
    r = 3
    for name, df in views.items():
        sg.cell(r, 1, name.replace("_", " ").replace("by ", "By ")).font = F_BOLD
        r += 1
        for j, h in enumerate(["Segment", "Applications", "Defaults", "Default rate"], 1):
            c = sg.cell(r, j, h)
            c.font, c.fill = F_HEAD, FILL_HEAD
        for row in df.itertuples(index=False):
            r += 1
            sg.cell(r, 1, row.segment).font = F_TXT
            sg.cell(r, 2, int(row.applications)).number_format = "#,##0"
            sg.cell(r, 3, int(row.defaults)).number_format = "#,##0"
            sg.cell(r, 4, f"=IFERROR(C{r}/B{r},0)").number_format = PCT
        r += 2
    sg.column_dimensions["A"].width = 32
    for col in "BCD":
        sg.column_dimensions[col].width = 14
    wb.save(path)


# --------------------------------------------------------------------------------------
# Power BI exports
# --------------------------------------------------------------------------------------
def powerbi_exports(pbi_dir: Path, *, scored: pd.DataFrame, cut_card, cut_gbm, views):
    pbi_dir.mkdir(parents=True, exist_ok=True)
    s = scored.copy()
    s["score_band"] = pd.cut(s["score"], bins=[-np.inf, 500, 525, 550, 575, 600, 625, np.inf],
                             labels=["<500", "500-524", "525-549", "550-574", "575-599", "600-624", "625+"]).astype(str)
    s["bureau_segment"] = np.where(s["has_bureau_history"] == 1, "Has bureau history", "Thin file")
    s.to_csv(pbi_dir / "scored_test_applications.csv", index=False)
    pd.concat([cut_card.assign(model="Scorecard"), cut_gbm.assign(model="Gradient boosting")]) \
        .to_csv(pbi_dir / "cutoff_strategy.csv", index=False)
    pd.concat([v.assign(view=k) for k, v in views.items()]).to_csv(pbi_dir / "portfolio_segments.csv", index=False)


def build_all(out_dir: Path, *, summary, iv, card, importance, cut_card, cut_gbm, views, scored,
              y_te, pd_card, pd_gbm, pd_gbm_raw):
    figures(out_dir / "figures", y_te=y_te, pd_card=pd_card, pd_gbm=pd_gbm, iv=iv, importance=importance,
            cut_card=cut_card, cut_gbm=cut_gbm, views=views, summary=summary)
    excel_model(out_dir / "cutoff_strategy_model.xlsx", cut_card=cut_card, cut_gbm=cut_gbm,
                summary=summary, views=views)
    powerbi_exports(out_dir / "powerbi", scored=scored, cut_card=cut_card, cut_gbm=cut_gbm, views=views)
    grid = np.linspace(0, 1, 51)
    roc_pts = {"fpr": grid}
    for name, p in (("scorecard", pd_card), ("gbm", pd_gbm)):
        fpr, tpr, _ = roc_curve(y_te, p)
        roc_pts[f"tpr_{name}"] = np.interp(grid, fpr, tpr)
    pd.DataFrame(roc_pts).round(4).to_csv(out_dir / "roc_points.csv", index=False)
    for k, v in views.items():
        v.to_csv(out_dir / f"segment_{k}.csv", index=False)
    (out_dir / "segments.json").write_text(json.dumps({k: v.to_dict("records") for k, v in views.items()}, indent=2))
