#!/bin/bash

BASE_URL="http://localhost:5000"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

PASS=0
FAIL=0

check() {
    local label=$1
    local status=$2
    local body=$3

    if [ "$status" -ge 200 ] && [ "$status" -lt 300 ]; then
        echo -e "${GREEN}✔ PASS${NC} [$status] $label"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}✘ FAIL${NC} [$status] $label"
        echo -e "   ${RED}→ $body${NC}"
        FAIL=$((FAIL + 1))
    fi
}

call_get() {
    local label=$1
    local endpoint=$2
    local response
    response=$(curl -s -o /tmp/api_body -w "%{http_code}" "$BASE_URL$endpoint")
    local body
    body=$(cat /tmp/api_body)
    check "$label" "$response" "$body"
}

call_post() {
    local label=$1
    local endpoint=$2
    local data=$3
    local response
    if [ -z "$data" ]; then
        response=$(curl -s -o /tmp/api_body -w "%{http_code}" -X POST "$BASE_URL$endpoint")
    else
        response=$(curl -s -o /tmp/api_body -w "%{http_code}" -X POST "$BASE_URL$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data")
    fi
    local body
    body=$(cat /tmp/api_body)
    check "$label" "$response" "$body"
}

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}       AUTO MPG — API TEST SUITE        ${NC}"
echo -e "${CYAN}========================================${NC}"

# ─── HEALTH ───────────────────────────────────────────
echo ""
echo -e "${YELLOW}--- HEALTH ---${NC}"
call_get  "GET  /health" "/health"

# ─── PIPELINE STEP BY STEP ───────────────────────────
echo ""
echo -e "${YELLOW}--- PIPELINE STEP BY STEP ---${NC}"
call_post "POST /load"        "/load"
call_post "POST /clean"       "/clean"
call_post "POST /select"      "/select"       '{"p": 0.75}'
call_post "POST /standardize" "/standardize"

# ─── PIPELINE COMPLETA ───────────────────────────────
echo ""
echo -e "${YELLOW}--- PIPELINE COMPLETA ---${NC}"
call_post "POST /pipeline" "/pipeline"

# ─── EDA ─────────────────────────────────────────────
echo ""
echo -e "${YELLOW}--- EDA ---${NC}"
call_get  "GET  /eda/info"                        "/eda/info"
call_get  "GET  /eda/outliers/horsepower"         "/eda/outliers/horsepower"
call_get  "GET  /eda/jarque_bera/horsepower"      "/eda/jarque_bera/horsepower"
call_get  "GET  /eda/normal_test/horsepower"      "/eda/normal_test/horsepower"
call_get  "GET  /eda/qqplot/horsepower"           "/eda/qqplot/horsepower"
call_get  "GET  /eda/boxplot/horsepower"          "/eda/boxplot/horsepower"
call_get  "GET  /eda/histograms"                  "/eda/histograms"

# ─── MODELLO ─────────────────────────────────────────
echo ""
echo -e "${YELLOW}--- MODELLO ---${NC}"
call_post "POST /model/train  (linear)"  "/model/train" '{"model_type": "linear"}'
call_post "POST /model/train  (ridge)"   "/model/train" '{"model_type": "ridge", "alpha": 0.5}'
call_post "POST /model/train  (lasso)"   "/model/train" '{"model_type": "lasso", "alpha": 0.5}'
call_post "POST /model/predict"          "/model/predict" \
    '{"displacement": 307, "cylinders": 8, "horsepower": 130, "weight": 3504, "acceleration": 12, "model_year": 70, "origin": 1}'
call_get  "GET  /model/coefficients"     "/model/coefficients"

# ─── RIEPILOGO ───────────────────────────────────────
echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "  ${GREEN}PASS: $PASS${NC}   ${RED}FAIL: $FAIL${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""