function formatMaybeDate(value) {
    if (!value) return "-";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString();
}

function formatSnapshotAge(minutes) {
    if (minutes === null || minutes === undefined) return "-";
    if (minutes < 1) return `${Math.round(minutes * 60)} sec`;
    if (minutes < 60) return `${minutes.toFixed(1)} min`;
    return `${(minutes / 60).toFixed(1)} h`;
}

function setBadge(activeState, subState) {
    const badge = document.getElementById("serviceBadge");
    badge.className = "pill";
    if (activeState === "active" && subState === "running") {
        badge.classList.add("ok");
        badge.textContent = "Running";
        return;
    }
    if (activeState === "activating") {
        badge.classList.add("warn");
        badge.textContent = "Starting";
        return;
    }
    badge.classList.add("bad");
    badge.textContent = `${activeState || "unknown"} / ${subState || "unknown"}`;
}

function fillRows(targetId, rows, emptyText, columns) {
    const body = document.getElementById(targetId);
    if (!rows || rows.length === 0) {
        body.innerHTML = `<tr><td colspan="${columns.length}">${emptyText}</td></tr>`;
        return;
    }
    body.innerHTML = rows.map((row) => {
        const cells = columns.map((col) => `<td>${row[col] ?? "-"}</td>`).join("");
        return `<tr>${cells}</tr>`;
    }).join("");
}

async function loadMonitorSummary() {
    const response = await fetch("/api/monitor/summary");
    const payload = await response.json();
    if (!payload.success) {
        throw new Error(payload.error || "Failed to load monitor summary");
    }

    const data = payload.data;
    const service = data.service;
    const botService = service.bot_service || {};
    const timer = service.refresh_timer || {};
    const runtime = service.runtime_status || {};
    const selector = runtime.selector_state || {};
    const snapshot = data.snapshot || {};
    const account = data.account || {};

    document.getElementById("serverTime").textContent = `Server UTC: ${formatMaybeDate(data.server_time)}`;
    document.getElementById("serviceState").textContent = `${botService.active_state || "-"} / ${botService.sub_state || "-"}`;
    document.getElementById("snapshotAge").textContent = formatSnapshotAge(snapshot.age_minutes);
    document.getElementById("openOrders").textContent = `${account.open_orders_count ?? 0}`;
    document.getElementById("activePositions").textContent = `${account.active_positions_count ?? 0}`;
    document.getElementById("startedAt").textContent = formatMaybeDate(botService.started_at);
    document.getElementById("mainPid").textContent = botService.main_pid || "-";
    document.getElementById("memoryCurrent").textContent = botService.memory_current || "-";
    document.getElementById("tasksCurrent").textContent = botService.tasks_current || "-";
    document.getElementById("lastSkipReason").textContent = selector.last_skip_reason || "-";
    document.getElementById("lastScanAt").textContent = formatMaybeDate(selector.last_scan_at);
    document.getElementById("nextRefresh").textContent = formatMaybeDate(timer.next_run);
    document.getElementById("lastRefresh").textContent = formatMaybeDate(timer.last_trigger);
    setBadge(botService.active_state, botService.sub_state);

    fillRows(
        "positionsBody",
        (account.active_positions || []).map((item) => ({
            symbol: item.symbol,
            side: item.side,
            size: item.size,
            unrealised_pnl: item.unrealised_pnl,
        })),
        "No active positions",
        ["symbol", "side", "size", "unrealised_pnl"],
    );

    fillRows(
        "ordersBody",
        (account.open_orders || []).map((item) => ({
            symbol: item.symbol,
            side: item.side,
            qty: item.qty,
            order_status: item.order_status,
        })),
        "No open orders",
        ["symbol", "side", "qty", "order_status"],
    );

    fillRows(
        "snapshotRows",
        (snapshot.rows || []).map((item) => ({
            symbol: item.symbol,
            edge_score: item.edge_score,
            win_rate_pct: item.win_rate_pct,
            net_pnl: item.net_pnl,
        })),
        "No snapshot rows",
        ["symbol", "edge_score", "win_rate_pct", "net_pnl"],
    );

    document.getElementById("botLogs").textContent = (service.recent_bot_logs || []).join("\n") || "No logs yet";
    document.getElementById("refreshLogs").textContent = (service.recent_refresh_logs || []).join("\n") || "No refresh logs yet";
}

async function refreshLoop() {
    try {
        await loadMonitorSummary();
    } catch (error) {
        document.getElementById("botLogs").textContent = error.message;
    }
}

refreshLoop();
setInterval(refreshLoop, 10000);
