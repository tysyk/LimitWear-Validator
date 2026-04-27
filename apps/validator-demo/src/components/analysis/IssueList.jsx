import { useState } from "react";
import { StatusBadge } from "./StatusBadge";

export function IssueList({ issues = [] }) {
  const [expanded, setExpanded] = useState(false);

  const visibleIssues = expanded ? issues : issues.slice(0, 3);

  return (
    <div className="issuesCard">
      <h3 className="issuesTitle">
        Issues Found <span className="countBubble">{issues.length}</span>
      </h3>

      {visibleIssues.map((v, i) => (
        <div key={i} className={`issueItem ${v.severity || "low"}`}>
          <div className="issueIcon">✕</div>

          <div>
            <strong>{v.title}</strong>
            <p>{v.message}</p>
          </div>

          <StatusBadge status={v.severity} small />
        </div>
      ))}

      {issues.length > 3 && (
        <button
          className="viewIssues"
          onClick={() => setExpanded((prev) => !prev)}
        >
          {expanded ? "Show less" : "View all issues →"}
        </button>
      )}
    </div>
  );
}