// Builds reports/credit_risk_recommendation.pptx from the pipeline outputs.
//   npm install pptxgenjs   (once)
//   node scripts/build_deck.js [reports]
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const dir = process.argv[2] || "reports";
const S = JSON.parse(fs.readFileSync(path.join(dir, "summary.json"), "utf8"));
const seg = JSON.parse(fs.readFileSync(path.join(dir, "segments.json"), "utf8"));
const csv = (f) => {
  const [h, ...rows] = fs.readFileSync(path.join(dir, f), "utf8").trim().split(/\r?\n/);
  const keys = h.split(",");
  return rows.map((r) => Object.fromEntries(r.split(",").map((v, i) => [keys[i], isNaN(+v) ? v : +v])));
};
const cutCard = csv("cutoff_scorecard.csv");
const cutGbm = csv("cutoff_gbm.csv");
const roc = csv("roc_points.csv");
// sparse category labels so the axis stays readable
const rocLabels = roc.map((r, i) => (i % 10 === 0 ? r.fpr.toFixed(1) : " "));
const cutLabels = cutCard.map((r, i) => (i % 2 === 1 ? `${Math.round(r.approval_rate * 100)}%` : " "));

// ---- numbers used in the narrative -------------------------------------------------
const m = S.models, st = S.strategy, d = S.data, ht = S.hypothesis_tests;
const best = st.scorecard_best, all = st.approve_all, gb = st.gbm_best;
const pct = (x, k = 1) => `${(x * 100).toFixed(k)}%`;
const uplift = (best.profit - all.profit) / all.profit;
const gbmExtra = (gb.profit - best.profit) / best.profit;
const fmtInt = (x) => Math.round(x).toLocaleString("en-US");

// ---- design tokens -----------------------------------------------------------------
const NAVY = "1E2761", ICE = "CADCFC", INK = "1A1A1A", MUTED = "595959", GRID = "E4E3DF";
const BLUE = "2A78D6", ORANGE = "EB6834", CARD = "F3F6FC", WHITE = "FFFFFF";
const HEAD = "Cambria", BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.author = "Vaibhav Barsaiyan";
pres.title = "Credit-risk scorecard and cut-off recommendation";
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

const FOOT = "Home Credit Default Risk (Kaggle) · held-out test set of " + fmtInt(d.test) + " loans · github.com/Vb-777";
pres.defineSlideMaster({
  title: "DARK",
  background: { color: NAVY },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.0, w: 11.7, h: 2.2,
    fontFace: HEAD, fontSize: 40, bold: true, color: WHITE, valign: "bottom", align: "left" }, text: "" } }],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: WHITE },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 1.0,
      fontFace: HEAD, fontSize: 26, bold: true, color: NAVY, valign: "top", align: "left" }, text: "" } },
    { text: { text: FOOT, options: { x: 0.6, y: 7.0, w: 10.5, h: 0.3, fontFace: BODY, fontSize: 10, color: MUTED } } },
  ],
  slideNumber: { x: 12.3, y: 7.0, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 10, color: MUTED },
});

const chartText = { catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontFace: "+mn-lt",
  valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
  valGridLine: { color: GRID, size: 0.75 }, catGridLine: { style: "none" }, titleFontFace: "+mn-lt",
  titleColor: INK, titleFontSize: 13, legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: INK };

function stat(slide, x, y, w, big, label, color = NAVY) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.45, fill: { color: CARD }, rectRadius: 0.08,
    line: { color: CARD }, objectName: "stat card" });
  slide.addText(big, { x: x + 0.25, y: y + 0.12, w: w - 0.5, h: 0.75, fontFace: HEAD, fontSize: 34, bold: true,
    color, margin: 0, isTextBox: true });
  slide.addText(label, { x: x + 0.25, y: y + 0.86, w: w - 0.5, h: 0.5, fontFace: BODY, fontSize: 13, color: MUTED,
    margin: 0, isTextBox: true });
}

// 1. Title ---------------------------------------------------------------------------
let s = pres.addSlide({ masterName: "DARK" });
s.addText(`Approving the best ${pct(best.approval_rate, 0)} of applicants raises portfolio profit by ${pct(uplift, 0)}`,
  { placeholder: "title" });
s.addText("Credit-risk scorecard and cut-off strategy on " + fmtInt(d.applications) +
  " consumer loans  ·  Vaibhav Barsaiyan, M.Tech CSE, IIT Hyderabad",
  { x: 0.8, y: 4.45, w: 11.7, h: 0.6, fontFace: BODY, fontSize: 18, color: ICE, isTextBox: true });
s.addNotes("Headline first: a score cut-off that rejects the riskiest fifth of applicants cuts the bad rate from " +
  pct(all.bad_rate) + " to " + pct(best.bad_rate) + " and raises profit by " + pct(uplift, 0) +
  " under the stated LGD and margin assumptions.");

// 2. Data and segments --------------------------------------------------------------
s = pres.addSlide({ masterName: "CONTENT" });
s.addText(`Thin-file applicants default ${pct(ht.thin_file_vs_bureau_history.relative_risk - 1, 0)} more often than those with a bureau record`,
  { placeholder: "title" });
stat(s, 0.6, 1.55, 3.6, fmtInt(d.applications), "loans, joined in SQL with " + "1.7M bureau records");
stat(s, 0.6, 3.15, 3.6, pct(d.default_rate), "portfolio default rate");
stat(s, 0.6, 4.75, 3.6, String(d.features_engineered), "features engineered (ratios, bureau history)");
const segRows = [...seg.by_bureau_history, ...seg.by_income_type];
s.addChart(pres.charts.BAR, [{ name: "Default rate", labels: segRows.map((r) => r.segment),
  values: segRows.map((r) => r.default_rate) }], {
  x: 4.6, y: 1.45, w: 8.1, h: 4.9, barDir: "bar", chartColors: [BLUE], showValue: true,
  dataLabelFormatCode: "0.0%", dataLabelColor: INK, dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt",
  valAxisLabelFormatCode: "0%", catAxisOrientation: "maxMin", showLegend: false, barGapWidthPct: 60,
  showTitle: true, title: "Default rate by segment", ...chartText });
s.addText(`Two-proportion z-test: z = ${ht.thin_file_vs_bureau_history.z.toFixed(1)}, p < 0.001, difference ` +
  `${pct(ht.thin_file_vs_bureau_history.ci_low)} to ${pct(ht.thin_file_vs_bureau_history.ci_high)} (95% CI)`,
  { x: 4.6, y: 6.4, w: 8.1, h: 0.4, fontFace: BODY, fontSize: 12, color: MUTED, isTextBox: true, margin: 0 });
s.addNotes("Thin-file applicants are the under-served segment that alternate data is meant to help. Chi-square tests " +
  "also reject independence of default from education (Cramer's V " + ht.education.cramers_v.toFixed(3) +
  ") and income type (V " + ht.income_type.cramers_v.toFixed(3) + "): significant but small effects.");

// 3. Models ----------------------------------------------------------------------------
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("Gradient boosting ranks risk better; the scorecard stays fully explainable", { placeholder: "title" });
s.addChart(pres.charts.LINE, [
  { name: "Scorecard", labels: rocLabels, values: roc.map((r) => r.tpr_scorecard) },
  { name: "Gradient boosting", labels: rocLabels, values: roc.map((r) => r.tpr_gbm) },
], { x: 0.6, y: 1.45, w: 6.4, h: 5.2, chartColors: [BLUE, ORANGE], lineSize: 2, lineDataSymbol: "none",
  showLegend: true, legendPos: "b", showTitle: true, title: "ROC curve (test set)",
  valAxisLabelFormatCode: "0%", valAxisMaxVal: 1, valAxisMinVal: 0, showCatAxisTitle: true, catAxisTitle: "False positive rate",
  showValAxisTitle: true, valAxisTitle: "True positive rate", catAxisTitleColor: MUTED, valAxisTitleColor: MUTED,
  catAxisTitleFontSize: 11, valAxisTitleFontSize: 11, ...chartText });
const hdr = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY } } });
const rows = [[hdr("Test-set metric"), hdr("Scorecard"), hdr("Gradient boosting")]];
for (const [k, lab, f] of [["auc", "AUC", (x) => x.toFixed(3)], ["gini", "Gini", (x) => x.toFixed(3)],
  ["ks", "KS", (x) => x.toFixed(3)], ["ece", "Calibration error", (x) => pct(x, 2)]]) {
  rows.push([lab, f(m.scorecard[k]), f(m.gbm_calibrated[k])]);
}
rows.push(["Characteristics", String(S.scorecard_characteristics.length), String(d.features_engineered)]);
s.addTable(rows, { x: 7.4, y: 1.55, w: 5.3, colW: [2.1, 1.6, 1.6], fontFace: BODY, fontSize: 14, color: INK,
  border: { type: "solid", pt: 0.75, color: GRID }, rowH: 0.45, align: "left" });
const ci = m.gbm_vs_scorecard;
s.addText([
  { text: "Bootstrap (" + ci.n_boot + " resamples): ", options: { bold: true } },
  { text: `boosting adds ${(ci.auc_diff).toFixed(3)} AUC (95% CI ${ci.ci_low.toFixed(3)} to ${ci.ci_high.toFixed(3)}).` },
], { x: 7.4, y: 4.45, w: 5.3, h: 0.8, fontFace: BODY, fontSize: 14, color: INK, isTextBox: true, margin: 0 });
s.addText([
  { text: "Scorecard: ", options: { bold: true } },
  { text: "WoE bins with monotonic risk, " + S.scorecard_characteristics.length +
    " characteristics, points per bin and reason codes for every decline." },
], { x: 7.4, y: 5.35, w: 5.3, h: 1.0, fontFace: BODY, fontSize: 14, color: INK, isTextBox: true, margin: 0 });

// 4. Cut-off ----------------------------------------------------------------------------
s = pres.addSlide({ masterName: "CONTENT" });
s.addText(`Profit peaks at an approval rate of ${pct(best.approval_rate, 0)}: score cut-off ${best.score_cutoff.toFixed(0)}`,
  { placeholder: "title" });
s.addChart(pres.charts.LINE, [
  { name: "Scorecard", labels: cutLabels, values: cutCard.map((r) => r.profit / 1e9) },
  { name: "Gradient boosting", labels: cutLabels, values: cutGbm.map((r) => r.profit / 1e9) },
], { x: 0.6, y: 1.45, w: 7.6, h: 5.2, chartColors: [BLUE, ORANGE], lineSize: 2, lineDataSymbol: "circle",
  lineDataSymbolSize: 6, showLegend: true, legendPos: "b", showTitle: true,
  title: "Test-set profit (billions, data currency) by approval rate", valAxisLabelFormatCode: "0.0",
  ...chartText });
stat(s, 8.6, 1.55, 4.1, `${pct(all.bad_rate)} → ${pct(best.bad_rate)}`, "bad rate of the approved book");
stat(s, 8.6, 3.15, 4.1, pct(best.bads_rejected_share, 0), "of defaults declined, while rejecting " + pct(1 - best.approval_rate, 0) + " of applicants");
stat(s, 8.6, 4.75, 4.1, "+" + pct(uplift, 0), "profit vs approving everyone", BLUE);
s.addText(`Assumes LGD ${pct(st.assumptions.lgd, 0)} and net margin ${pct(st.assumptions.net_margin, 0)}; both are editable in the Excel model.`,
  { x: 8.6, y: 6.3, w: 4.1, h: 0.5, fontFace: BODY, fontSize: 11, color: MUTED, isTextBox: true, margin: 0 });

// 5. Recommendation ------------------------------------------------------------------
s = pres.addSlide({ masterName: "DARK" });
s.addText("Recommendation", { x: 0.8, y: 0.6, w: 11.7, h: 0.9, fontFace: HEAD, fontSize: 36, bold: true, color: WHITE, isTextBox: true });
const recs = [
  ["1", `Deploy the scorecard with a cut-off of ${best.score_cutoff.toFixed(0)}`,
    `Approves ${pct(best.approval_rate, 0)}, bad rate ${pct(best.bad_rate)}, profit +${pct(uplift, 0)}; every decline carries reason codes.`],
  ["2", "Run gradient boosting as the challenger",
    `It earns a further ${pct(gbmExtra)} profit at the same approval rate; promote it once explanations pass model-risk review.`],
  ["3", "Validate before go-live",
    "Replace LGD and margin with finance figures, add an out-of-time test, and apply reject inference to declined applicants."],
];
recs.forEach(([n, head, body], i) => {
  const y = 1.9 + i * 1.6;
  s.addShape(pres.shapes.OVAL, { x: 0.8, y, w: 0.75, h: 0.75, fill: { color: ICE }, line: { color: ICE }, objectName: "step marker" });
  s.addText(n, { x: 0.8, y, w: 0.75, h: 0.75, align: "center", valign: "middle", fontFace: HEAD, fontSize: 24, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  s.addText(head, { x: 1.85, y: y - 0.05, w: 10.6, h: 0.5, fontFace: HEAD, fontSize: 22, bold: true, color: WHITE, isTextBox: true, margin: 0 });
  s.addText(body, { x: 1.85, y: y + 0.48, w: 10.6, h: 0.6, fontFace: BODY, fontSize: 16, color: ICE, isTextBox: true, margin: 0 });
});

const out = path.join(dir, "credit_risk_recommendation.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("wrote " + out));
