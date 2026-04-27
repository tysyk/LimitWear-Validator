export function StatusBadge({ status, small }) {
  const map = {
    PASS: "green",
    WARN: "orange",
    FAIL: "red",
    NEED_REVIEW: "gray",
    high: "red",
    medium: "orange",
    low: "gray",
  };

  return (
    <span className={`badge ${map[status]} ${small ? "small" : ""}`}>
      {status}
    </span>
  );
}