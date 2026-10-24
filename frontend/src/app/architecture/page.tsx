import { MermaidDiagram } from "@/components/MermaidDiagram";
import { AblationChart } from "@/components/AblationChart";

const SYSTEM_DIAGRAM = `flowchart TD
    subgraph HF["Hugging Face Hub"]
        DS[("ml-latest-small\\nratings, movies, tags, links")]
    end

    subgraph NB["notebooks/ (learning spine)"]
        N1["01 Explore MovieLens"] --> N2["02 NVTabular features\\n(context, user-take)"]
        N2 --> N3["03 Item text embeddings\\n(sentence-transformer)"]
        N3 --> N4["04 Two-Tower + cold-start\\n+ ablation"]
    end

    subgraph Modal["Modal GPU - training/modal_app.py"]
        WF["NVTabular Workflow\\nfit + transform"]
        EMB["Sentence-transformer\\nitem text embeddings"]
        TT["Merlin Models TwoTowerModel\\nin-batch negative sampling\\n+ cold-start training"]
        EVAL["Offline eval:\\nRecall@K / NDCG@K\\n+ ablation"]
    end

    subgraph ART["artifacts/ (committed to repo)"]
        QT["Query tower weights (.npz)"]
        IE["Precomputed item embeddings"]
        WFA["feature_vocab.json"]
        METRICS["eval_metrics.json"]
    end

    subgraph BE["backend/ - FastAPI (CPU only)"]
        REC["POST /recommend"]
        MET["GET /eval/metrics"]
    end

    subgraph FE["frontend/ - Next.js"]
        UI1["User picker + context"]
        UI2["Cold-start simulator"]
        UI3["Architecture + ablation"]
    end

    DS --> N1
    N4 -. proves out .-> WF
    DS --> WF --> EMB --> TT --> EVAL
    WF --> WFA
    TT --> QT
    TT --> IE
    EVAL --> METRICS

    WFA --> REC
    QT --> REC
    IE --> REC
    METRICS --> MET

    REC --> UI1
    REC --> UI2
    MET --> UI3
`;

const TOWER_DIAGRAM = `flowchart LR
    subgraph REQ["Incoming request"]
        U["user_id\\n(or recent_movie_ids\\nfor cold-start)"]
        CTX["Context:\\nday_of_week, time_of_day,\\nsimulated device"]
    end

    subgraph QT["Query tower - runs on every request"]
        UF["User features:\\nid, user-take sequence, context"]
        QEMB["Dense layers\\n-> user embedding"]
    end

    subgraph IT["Item tower - runs once, at training time"]
        ITF["Item features:\\nid, genres,\\ntext embedding (frozen)"]
        IEMB[("Precomputed item embeddings\\n- one per movie, cached")]
    end

    U --> UF
    CTX --> UF
    UF --> QEMB
    ITF --> ITT["Dense layers"] --> IEMB

    QEMB --> DOT["Dot product\\nvs every cached item embedding"]
    IEMB --> DOT
    DOT --> SORT["Sort descending"]
    SORT --> TOPK["Top-K recommendations"]
`;

const SCALE_ROWS: [string, string, string, string][] = [
  ["Catalog size", "~9,700 movies", "tens of thousands of titles/episodes", "hundreds of millions of items"],
  ["Users", "~610", "millions", "billions"],
  [
    "Item-side retrieval",
    "exhaustive dot product, in-process",
    "precomputed embeddings, likely still exhaustive or light filtering",
    "mandatory ANN (FAISS/ScaNN) + multi-stage retrieve-then-rank funnel",
  ],
  [
    "Query tower serving",
    "plain numpy inside FastAPI, CPU, single instance",
    "Triton Inference Server, autoscaled, low-latency SLA",
    "Triton/custom serving, GPU, heavy sharding + caching",
  ],
  [
    "Retrain cadence",
    "manual, on-demand (one Modal run)",
    "daily batch retrain, hot-reloaded without downtime",
    "continuous/near-real-time online updates",
  ],
  [
    "Cold-start handling",
    "user-take + dropped-user-id training trick",
    "same technique, at scale",
    "same, plus exploration/bandit strategies",
  ],
];

export default function ArchitecturePage() {
  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <h1 className="text-2xl font-semibold tracking-tight text-neutral-50">Architecture</h1>
      <p className="mt-2 max-w-3xl text-sm text-neutral-400">
        How this project is actually put together, end to end — not just the two-tower model,
        but the full path from a Hugging Face dataset to a page you can click around in.
      </p>

      <section className="mt-8">
        <h2 className="mb-3 text-lg font-medium text-neutral-100">End-to-end system</h2>
        <MermaidDiagram chart={SYSTEM_DIAGRAM} id="system-diagram" />
      </section>

      <section className="mt-10">
        <h2 className="mb-3 text-lg font-medium text-neutral-100">
          Two-tower serving mechanics
        </h2>
        <p className="mb-3 text-sm text-neutral-400">
          The thing that makes this architecture fast enough for real-time, context-aware
          recommendations: the item tower only ever runs at training time, producing a
          precomputed embedding per movie; the query tower is the only thing that runs per
          request.
        </p>
        <MermaidDiagram chart={TOWER_DIAGRAM} id="tower-diagram" />
      </section>

      <section className="mt-10">
        <h2 className="mb-3 text-lg font-medium text-neutral-100">
          Ablation: does the enrichment earn its complexity?
        </h2>
        <p className="mb-4 text-sm text-neutral-400">
          Three variants, same data, same eval protocol: a plain two-tower model (ids + genres
          only), the enriched model (+ context, + user-take, + frozen text embeddings, +
          cold-start training), and that same enriched model with its identity forced to
          &quot;unknown&quot; at evaluation time.
        </p>
        <AblationChart />
      </section>

      <section className="mt-10">
        <h2 className="mb-3 text-lg font-medium text-neutral-100">Scale tradeoffs</h2>
        <p className="mb-4 text-sm text-neutral-400">
          This demo intentionally runs at MovieLens scale. Here&apos;s an honest accounting of
          what would need to change at a mid-size streaming platform&apos;s production scale
          and at hyperscale.
        </p>
        <div className="overflow-x-auto rounded-lg border border-neutral-800">
          <table className="w-full text-left text-sm">
            <thead className="bg-neutral-900 text-neutral-400">
              <tr>
                <th className="px-3 py-2 font-medium">Axis</th>
                <th className="px-3 py-2 font-medium">This demo</th>
                <th className="px-3 py-2 font-medium">Mid-size streaming platform</th>
                <th className="px-3 py-2 font-medium">Hyperscale</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800">
              {SCALE_ROWS.map((row) => (
                <tr key={row[0]} className="text-neutral-300">
                  {row.map((cell, i) => (
                    <td
                      key={i}
                      className={`px-3 py-2 align-top ${i === 0 ? "font-medium text-neutral-200" : "text-neutral-400"}`}
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mt-10">
        <h2 className="mb-3 text-lg font-medium text-neutral-100">How this was built</h2>
        <p className="text-sm text-neutral-400">
          Learning-first: <code className="text-neutral-300">notebooks/</code> is a tutorial
          sequence (exploration → feature engineering → text embeddings → the two-tower model
          + ablation), and only once a technique is proven out there does it get promoted into{" "}
          <code className="text-neutral-300">training/</code>,{" "}
          <code className="text-neutral-300">backend/</code>, and{" "}
          <code className="text-neutral-300">frontend/</code>. My own reasoning through the
          tradeoffs along the way is recorded in{" "}
          <code className="text-neutral-300">docs/ARCHITECTURE_DECISIONS.md</code>.
        </p>
      </section>

      <section className="mt-10 border-t border-neutral-800 pt-6 text-sm text-neutral-500">
        <h2 className="mb-2 text-base font-medium text-neutral-300">Credits</h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>
            Inspiration for the specific technique set (context features, user-take sequences,
            item text embeddings, in-batch negative sampling):{" "}
            <a
              className="underline hover:text-neutral-300"
              href="https://medium.com/tech-p7s1/video-recommendations-at-joyn-two-tower-or-not-to-tower-that-was-never-a-question-6c6f182ade7c"
            >
              Joyn Tech, &quot;Video Recommendations at Joyn&quot;
            </a>
            .
          </li>
          <li>
            Framework:{" "}
            <a className="underline hover:text-neutral-300" href="https://developer.nvidia.com/merlin">
              NVIDIA Merlin
            </a>
            .
          </li>
          <li>
            Dataset: MovieLens <code>ml-latest-small</code>, via a Hugging Face Hub mirror.
          </li>
        </ul>
      </section>
    </div>
  );
}
