/** A response may update visible state only while it owns the latest request. */
export function createLatestRequestGate() {
  let generation=0;
  return {
    begin() {
      const request=++generation;
      return { isCurrent:()=>request===generation };
    },
    invalidate() { generation++; },
  };
}
