#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash fetch_papers.sh [output_dir]
#
# Example:
#   bash fetch_papers.sh papers
#
# Notes:
# - This script uses curl with redirect following and retries.
# - Some ACM / IEEE / ScienceDirect links may require institutional access,
#   cookies, or may return HTML instead of a PDF. The script checks the
#   Content-Type when possible and warns on suspicious downloads.

OUTDIR="${1:-papers}"
OUTDIR="${OUTDIR%/}"
mkdir -p "$OUTDIR"

# Common curl flags:
# -L: follow redirects
# -f: fail on HTTP errors
# --retry / --retry-all-errors: retry transient failures
# --connect-timeout / --max-time: avoid hanging forever
CURL_FLAGS=(
  -L
  -f
  --http1.1
  --compressed
  --retry 4
  --retry-delay 2
  --retry-all-errors
  --connect-timeout 20
  --max-time 300
  -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
  -H "Accept: application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
  -H "Accept-Language: en-US,en;q=0.9"
)

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

parse_entry() {
  local line="$1"
  local parsed_name=""
  local parsed_url=""

  if [[ "$line" == *$'\t'* ]]; then
    parsed_name="${line%%$'\t'*}"
    parsed_url="${line#*$'\t'}"
  else
    parsed_url="${line##*[[:space:]]}"
    parsed_name="${line%"$parsed_url"}"
    parsed_name="$(trim "$parsed_name")"
    parsed_name="${parsed_name%—}"
    parsed_name="${parsed_name%-}"
    parsed_name="$(trim "$parsed_name")"
  fi

  PARSED_NAME="$(trim "$parsed_name")"
  PARSED_URL="$(trim "$parsed_url")"
}

is_pdf_file() {
  local path="$1"
  local header=""
  local mime=""

  [[ -s "$path" ]] || return 1

  header="$(head -c 5 "$path" 2>/dev/null || true)"
  if [[ "$header" == "%PDF-" ]]; then
    return 0
  fi

  mime="$(file --brief --mime-type "$path" 2>/dev/null || true)"
  [[ "$mime" == "application/pdf" ]]
}

download_candidate() {
  local url="$1"
  local tmpfile="$2"
  local referer="${3:-}"

  rm -f "$tmpfile"
  if [[ -n "$referer" ]]; then
    curl "${CURL_FLAGS[@]}" -e "$referer" -o "$tmpfile" "$url"
  else
    curl "${CURL_FLAGS[@]}" -o "$tmpfile" "$url"
  fi
}

download() {
  local name="$1"
  local url="$2"
  local outfile="$OUTDIR/$name.pdf"
  local tmpfile="${outfile}.part"
  local -a candidate_urls=()
  local -a candidate_referrers=()
  local doi=""
  local arnumber=""
  local i=0

  echo "==> Downloading: $name"
  echo "    URL: $url"

  candidate_urls+=("$url")
  candidate_referrers+=("")

  if [[ "$url" == *"dl.acm.org/doi/"* ]]; then
    doi="${url#*dl.acm.org/doi/}"
    doi="${doi#pdf/}"
    doi="${doi#epdf/}"
    doi="${doi%%\?*}"
    candidate_urls+=("https://dl.acm.org/doi/pdf/${doi}?download=true")
    candidate_referrers+=("https://dl.acm.org/doi/${doi}")
    candidate_urls+=("https://dl.acm.org/doi/epdf/${doi}?download=true")
    candidate_referrers+=("https://dl.acm.org/doi/${doi}")
    candidate_urls+=("https://dl.acm.org/doi/pdf/${doi}")
    candidate_referrers+=("https://dl.acm.org/doi/${doi}")
  elif [[ "$url" == *"ieeexplore.ieee.org"* ]] && [[ "$url" == *"arnumber="* ]]; then
    arnumber="${url##*arnumber=}"
    arnumber="${arnumber%%[^0-9]*}"
    if [[ -n "$arnumber" ]]; then
      candidate_urls+=("https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=${arnumber}")
      candidate_referrers+=("https://ieeexplore.ieee.org/document/${arnumber}")
      candidate_urls+=("https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=${arnumber}")
      candidate_referrers+=("https://ieeexplore.ieee.org/document/${arnumber}")
    fi
  fi

  for (( i=0; i<${#candidate_urls[@]}; i++ )); do
    if download_candidate "${candidate_urls[$i]}" "$tmpfile" "${candidate_referrers[$i]}"; then
      if is_pdf_file "$tmpfile"; then
        mv "$tmpfile" "$outfile"
        echo "    Saved to: $outfile"
        return 0
      fi

      local mime
      mime="$(file --brief --mime-type "$tmpfile" 2>/dev/null || true)"
      echo "    Warning: non-PDF response (MIME '$mime') from: ${candidate_urls[$i]}" >&2
    fi
  done

  rm -f "$tmpfile"
  echo "    Failed: $name" >&2
  return 1
}

# Keep going even if some papers fail.
failures=0

while IFS= read -r line; do
  line="$(trim "$line")"
  [[ -z "${line:-}" ]] && continue
  [[ "$line" == \#* ]] && continue

  parse_entry "$line"
  if [[ -z "${PARSED_NAME:-}" ]] || [[ -z "${PARSED_URL:-}" ]]; then
    echo "    Failed to parse entry: $line" >&2
    failures=$((failures + 1))
    continue
  fi

  if ! download "$PARSED_NAME" "$PARSED_URL"; then
    failures=$((failures + 1))
  fi
done <<'EOF'
Remy_2013	https://web.mit.edu/remy/TCPexMachina.pdf
PCC_2015	https://www.usenix.org/system/files/conference/nsdi15/nsdi15-paper-dong.pdf
PCC_Vivace_2018	https://www.usenix.org/system/files/conference/nsdi18/nsdi18-dong.pdf
Sprout_2013	https://www.usenix.org/system/files/conference/nsdi13/nsdi13-final113.pdf
TIMELY_2015	https://conferences.sigcomm.org/sigcomm/2015/pdf/papers/p537.pdf
DCQCN_2015	https://conferences.sigcomm.org/sigcomm/2015/pdf/papers/p523.pdf
HPCC_2019	https://dl.acm.org/doi/pdf/10.1145/3341302.3342085
PCC_RL_Aurora_2019	https://proceedings.mlr.press/v97/jay19a/jay19a.pdf
Orca_2020	https://dl.acm.org/doi/pdf/10.1145/3387514.3405892
Spine_2022	https://dl.acm.org/doi/pdf/10.1145/3555050.3569125
Sage_2023	https://dl.acm.org/doi/pdf/10.1145/3603269.3604838
Mutant_2025	https://www.usenix.org/system/files/nsdi25-pappone.pdf
Mowgli_2025	https://www.usenix.org/system/files/nsdi25-agarwal.pdf
Comyco	https://dl.acm.org/doi/pdf/10.1145/3343031.3351014
Vibra	https://dl.acm.org/doi/pdf/10.1145/3458306.3460993
Improving_Generalization_for_Neural_Adaptive_Video_Streaming_via_Meta_Reinforcement_Learning	https://dl.acm.org/doi/pdf/10.1145/3503161.3548331
MetaABR	https://dl.acm.org/doi/pdf/10.1145/3565477.3569155
GreenABR	https://dl.acm.org/doi/10.1145/3524273.3528188
Index_Aware_Reinforcement_Learning_for_Adaptive_Video_Streaming	https://dl.acm.org/doi/10.1145/3492866.3549726
DRALVS https://dl.acm.org/doi/pdf/10.1145/3690931.3690961
QoE_Aware_Adaptive_Bitrate_Algorithm_Based_on_Subepisodic_Deep_Reinforcement_Learning_for_DASH	https://dl.acm.org/doi/10.1145/3587716.3587733
Incendio	https://dl.acm.org/doi/pdf/10.1145/3592473.3592564
Jade	https://dl.acm.org/doi/pdf/10.1145/3581783.3611771
Bitrate_Adaptation_and_Guidance_With_Meta_Reinforcement_Learning	https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10443925
Pensieve_5G	https://arxiv.org/pdf/2212.14479
FIRM — https://www.usenix.org/system/files/osdi20-qiu.pdf
AWARE — https://www.usenix.org/system/files/atc23-qiu-haoran.pdf
Decentralized Application-Level Adaptive Scheduling for Multi-Instance DNNs — https://www.usenix.org/system/files/atc23-sung.pdf
Intelligent Resource Scheduling for Co-located Latency-critical Services — https://www.usenix.org/system/files/fast23-liu.pdf
Network Contention-Aware Cluster Scheduling with Reinforcement Learning — https://arxiv.org/pdf/2310.20209
AWRE https://www.usenix.org/system/files/atc23-qiu-haoran.pdf
SOL https://www.microsoft.com/en-us/research/wp-content/uploads/2022/01/On-Node-Learning-ASPLOS22.pdf
GreenDRL https://dl.acm.org/doi/pdf/10.1145/3542929.3563501
EOF

echo
if [[ "$failures" -eq 0 ]]; then
  echo "All downloads completed."
else
  echo "$failures download(s) failed or were inaccessible." >&2
  exit 1
fi
