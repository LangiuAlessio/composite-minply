#!/usr/bin/env bash
# Run an experiment of this bundle on a remote compute host instead of the laptop.
#
# The FE jobs are small but relentless, and there is no reason to cook the Mac's battery with
# them. With the same ccx version the numbers are the same: exp9 reproduces on both hosts to the
# last printed digit, which is itself worth knowing for a paper about reproducibility.
#
#   ./run_on_head.sh experiments.exp9_experimental_validation
#   ./run_on_head.sh experiments.exp6_haftka_walsh
#
# Results are copied back into code/experiments/_out/ so the local tree stays authoritative.
set -euo pipefail

MODULE="${1:?usage: run_on_head.sh <python.module.path>}"
# Host su cui girare, e directory remota. Erano cablati sull'host di casa: parametrizzati il
# 2026-08-26 per il rilascio pubblico, perche' un tool che si pubblica non deve portarsi dietro
# la topologia di rete di chi lo ha scritto.
REMOTE="${CCX_REMOTE:?set CCX_REMOTE to an ssh host with CalculiX installed, e.g. CCX_REMOTE=mybox}"
# ⚠️ Niente apici dentro ${:-}: restavano LETTERALI, la tilde non veniva espansa dal remoto e il
# `cd` falliva per chiunque non impostasse CCX_REMOTE_DIR -- cioe' col default, sempre.
RDIR="${CCX_REMOTE_DIR:-composite-opt-code}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> sync -> ${REMOTE}"
rsync -az --delete --exclude '_out' --exclude '__pycache__' "${HERE}/" "${REMOTE}:${RDIR}/"

echo "==> run ${MODULE} on ${REMOTE}"
ssh "${REMOTE}" "cd ${RDIR} && CCX_BIN=/usr/bin/ccx python3 -m ${MODULE}"

echo "==> fetch results"
rsync -az "${REMOTE}:${RDIR}/experiments/_out/" "${HERE}/experiments/_out/" 2>/dev/null || true
echo "done: results in code/experiments/_out/"
