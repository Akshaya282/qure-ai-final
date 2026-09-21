export default function BatchResultsTable({ batchResult }) {
  if (!batchResult) return null

  return (
    <div className="batch-results">
      <h4 className="fc-heading">
        Batch results: {batchResult.n_succeeded} of {batchResult.n_rows} rows predicted
      </h4>
      <div className="comparison-table-wrap">
        <table className="comparison-table">
          <thead>
            <tr>
              <th>Row</th>
              <th>Probability</th>
              <th>Prediction</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {batchResult.results.map((r) => (
              <tr key={r.row_index}>
                <td>{r.row_index + 1}</td>
                {r.error ? (
                  <td colSpan={3} className="batch-row-error">Error: {r.error}</td>
                ) : (
                  <>
                    <td>{(r.probability * 100).toFixed(1)}%</td>
                    <td>{r.prediction}</td>
                    <td>{r.confidence_band}</td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
