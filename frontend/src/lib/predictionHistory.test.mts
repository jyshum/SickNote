import test from "node:test";
import assert from "node:assert/strict";
import { addPredictionToHistory, type PredictionHistoryItem } from "./predictionHistory.ts";

const baseItem: PredictionHistoryItem = {
  id: "first",
  label: "healthy",
  confidence: 0.82,
  source: "recorded",
  createdAt: "2026-05-30T12:00:00.000Z",
  audioUrl: "blob:first",
};

test("addPredictionToHistory puts the newest prediction first and caps the list at five", () => {
  const existing: PredictionHistoryItem[] = [
    baseItem,
    { ...baseItem, id: "second" },
    { ...baseItem, id: "third" },
    { ...baseItem, id: "fourth" },
    { ...baseItem, id: "fifth" },
  ];

  const next = addPredictionToHistory(existing, {
    label: "abnormal",
    confidence: 0.734,
    source: "uploaded",
    audioUrl: "blob:new",
    createdAt: "2026-05-30T12:05:00.000Z",
  });

  assert.equal(next.length, 5);
  assert.equal(next[0].label, "abnormal");
  assert.equal(next[0].confidence, 0.734);
  assert.equal(next[0].source, "uploaded");
  assert.equal(next[0].audioUrl, "blob:new");
  assert.equal(next[0].createdAt, "2026-05-30T12:05:00.000Z");
  assert.match(next[0].id, /^uploaded-2026-05-30T12:05:00\.000Z-/);
  assert.deepEqual(
    next.map((item) => item.id),
    [next[0].id, "first", "second", "third", "fourth"],
  );
});
