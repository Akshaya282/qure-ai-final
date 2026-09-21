import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts'

const MODEL_COLORS = {
  LogisticRegression: '#8884d8',
  RandomForest: '#3A7563',
  SVM_RBF: '#A6772C',
  ShallowMLP: '#57645D',
  HybridVQC: '#5B4636',
  QSVM: '#A23B33',
}

/**
 * Merges each model's {fpr, tpr} arrays onto a shared set of FPR points so Recharts can plot
 * every model as its own line on one chart. Models were computed with independent thresholds,
 * so we resample each onto a common 0..1 FPR grid via simple linear interpolation.
 */
function mergeRocCurves(modelsPerf) {
  const grid = Array.from({ length: 41 }, (_, i) => i / 40) // 0, 0.025, ..., 1.0
  const modelNames = Object.keys(modelsPerf)

  return grid.map((fprPoint) => {
    const row = { fpr: fprPoint }
    modelNames.forEach((name) => {
      const { fpr, tpr } = modelsPerf[name].roc_curve
      row[name] = interpolateTpr(fpr, tpr, fprPoint)
    })
    return row
  })
}

function interpolateTpr(fprArr, tprArr, x) {
  if (!fprArr || fprArr.length === 0) return null
  for (let i = 0; i < fprArr.length - 1; i++) {
    if (x >= fprArr[i] && x <= fprArr[i + 1]) {
      const span = fprArr[i + 1] - fprArr[i]
      if (span === 0) return tprArr[i]
      const t = (x - fprArr[i]) / span
      return tprArr[i] + t * (tprArr[i + 1] - tprArr[i])
    }
  }
  return x <= fprArr[0] ? tprArr[0] : tprArr[tprArr.length - 1]
}

export default function RocChart({ modelsPerf, highlightModel }) {
  const data = mergeRocCurves(modelsPerf)
  const modelNames = Object.keys(modelsPerf)

  return (
    <ResponsiveContainer width="100%" height={340}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#DEDAD0" />
        <XAxis
          dataKey="fpr"
          type="number"
          domain={[0, 1]}
          tickFormatter={(v) => v.toFixed(1)}
          label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -4, fontSize: 12 }}
          fontSize={11}
        />
        <YAxis
          domain={[0, 1]}
          tickFormatter={(v) => v.toFixed(1)}
          label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fontSize: 12 }}
          fontSize={11}
        />
        <Tooltip formatter={(value) => (value == null ? '-' : value.toFixed(3))} />
        <Legend wrapperStyle={{ fontSize: '12px' }} />
        {/* Diagonal reference line: AUC = 0.5, a random-guess classifier */}
        <Line
          data={[{ fpr: 0, chance: 0 }, { fpr: 1, chance: 1 }]}
          dataKey="chance"
          stroke="#DEDAD0"
          strokeDasharray="4 4"
          dot={false}
          name="Random Guess"
          isAnimationActive={false}
        />
        {modelNames.map((name) => (
          <Line
            key={name}
            type="monotone"
            dataKey={name}
            stroke={MODEL_COLORS[name] || '#999'}
            strokeWidth={name === highlightModel ? 3 : 1.5}
            dot={false}
            name={`${name} (AUC ${modelsPerf[name].auc_roc?.toFixed(3) ?? 'N/A'})`}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
