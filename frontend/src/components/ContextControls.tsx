"use client";

import { DAYS_OF_WEEK, DEVICES, TIMES_OF_DAY } from "@/lib/api";

export type Context = {
  dayOfWeek: string;
  timeOfDay: string;
  device: string;
};

export function ContextControls({
  value,
  onChange,
}: {
  value: Context;
  onChange: (next: Context) => void;
}) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
      <label className="flex flex-col gap-1 text-xs text-neutral-400">
        Day of week
        <select
          className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm text-neutral-100"
          value={value.dayOfWeek}
          onChange={(e) => onChange({ ...value, dayOfWeek: e.target.value })}
        >
          {DAYS_OF_WEEK.map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
      </label>
      <label className="flex flex-col gap-1 text-xs text-neutral-400">
        Time of day
        <select
          className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm text-neutral-100"
          value={value.timeOfDay}
          onChange={(e) => onChange({ ...value, timeOfDay: e.target.value })}
        >
          {TIMES_OF_DAY.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </label>
      <label className="flex flex-col gap-1 text-xs text-neutral-400">
        Device{" "}
        <span className="text-neutral-600" title="MovieLens has no device field — this does not change the model's output, see Architecture page">
          (illustrative only)
        </span>
        <select
          className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm text-neutral-100"
          value={value.device}
          onChange={(e) => onChange({ ...value, device: e.target.value })}
        >
          {DEVICES.map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
