// Bybit Trading Bot - Web UI JavaScript

// ============= Global State =============
let socket = null;
let performanceChart = null;
let backtestChart = null;

// ============= Initialization =============
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeWebSocket();
    initializeCharts();
    loadConfiguration();
    setupEventListeners();
    updateStatus();

    // Auto-refresh status every 5 seconds
    setInterval(updateStatus, 5000);
});

// ============= Tab Navigation =============
function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.dataset.tab;

            // Remove active class from all
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked
            button.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });
}

// ============= WebSocket Connection =============
function initializeWebSocket() {
    socket = io();

    socket.on('connect', () => {
        console.log('WebSocket connected');
        socket.emit('subscribe_logs');
        socket.emit('subscribe_status');
    });

    socket.on('disconnect', () => {
        console.log('WebSocket disconnected');
    });

    socket.on('logs_update', (data) => {
        updateLogs(data.logs);
    });

    socket.on('status_update', (data) => {
        updateStatusFromWebSocket(data);
    });
}

// ============= Charts =============
function initializeCharts() {
    // Performance Chart
    const perfCtx = document.getElementById('performanceChart');
    if (perfCtx) {
        performanceChart = new Chart(perfCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Balance',
                    data: [],
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        ticks: { color: '#94a3b8' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        ticks: { color: '#94a3b8' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }
}

// ============= Configuration =============
async function loadConfiguration() {
    try {
        const response = await fetch('/api/config');
        const data = await response.json();

        if (data.success) {
            const config = data.config;

            // API Settings
            document.getElementById('apiKey').value = config.api?.api_key || '';
            document.getElementById('apiSecret').value = config.api?.api_secret || '';
            document.getElementById('testnet').checked = config.api?.testnet || false;

            // Trading Settings
            document.getElementById('symbol').value = config.trading?.symbol || 'ETHUSDT';
            document.getElementById('timeframe').value = config.strategy?.timeframe || 5;
            document.getElementById('fixedSize').value = config.trading?.fixed_usdt_size || 2.0;
            document.getElementById('dryRun').checked = config.trading?.dry_run !== false;

            // Strategy Parameters
            document.getElementById('emaFast').value = config.strategy?.ema_fast || 9;
            document.getElementById('emaSlow').value = config.strategy?.ema_slow || 21;
            document.getElementById('atrPeriod').value = config.strategy?.atr_period || 14;
            document.getElementById('slAtrMult').value = config.strategy?.sl_atr_multiplier || 3.0;
            document.getElementById('riskReward').value = config.strategy?.risk_reward_ratio || 1.5;

            // Risk Management
            document.getElementById('maxTrades').value = config.risk?.max_trades_per_day || 10;
            document.getElementById('maxLoss').value = config.risk?.max_daily_loss_usdt || 10.0;
        }
    } catch (error) {
        console.error('Error loading configuration:', error);
        showNotification('Failed to load configuration', 'error');
    }
}

async function saveConfiguration() {
    try {
        const config = {
            api: {
                api_key: document.getElementById('apiKey').value,
                api_secret: document.getElementById('apiSecret').value,
                testnet: document.getElementById('testnet').checked
            },
            trading: {
                symbol: document.getElementById('symbol').value,
                category: 'linear',
                max_leverage: 10,
                fixed_usdt_size: parseFloat(document.getElementById('fixedSize').value),
                dry_run: document.getElementById('dryRun').checked
            },
            strategy: {
                timeframe: parseInt(document.getElementById('timeframe').value),
                ema_fast: parseInt(document.getElementById('emaFast').value),
                ema_slow: parseInt(document.getElementById('emaSlow').value),
                atr_period: parseInt(document.getElementById('atrPeriod').value),
                risk_reward_ratio: parseFloat(document.getElementById('riskReward').value),
                sl_atr_multiplier: parseFloat(document.getElementById('slAtrMult').value),
                volume_ma_period: 20,
                trading_start_hour: 8,
                trading_end_hour: 20,
                volume_multiplier: 1.2,
                volume_lookback: 20,
                levels_lookback: 10
            },
            risk: {
                max_trades_per_day: parseInt(document.getElementById('maxTrades').value),
                max_daily_loss_usdt: parseFloat(document.getElementById('maxLoss').value)
            }
        };

        const response = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const data = await response.json();

        if (data.success) {
            showNotification('Configuration saved successfully!', 'success');
        } else {
            showNotification('Failed to save configuration: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error saving configuration:', error);
        showNotification('Failed to save configuration', 'error');
    }
}

// ============= Backtest =============
async function runBacktest() {
    try {
        const params = {
            days: parseInt(document.getElementById('backtestDays').value),
            initial_balance: parseFloat(document.getElementById('backtestBalance').value),
            fixed_size: parseFloat(document.getElementById('backtestSize').value),
            win_rate: parseFloat(document.getElementById('backtestWinRate').value)
        };

        showNotification('Running backtest...', 'info');

        const response = await fetch('/api/backtest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });

        const data = await response.json();

        if (data.success) {
            displayBacktestResults(data);
            showNotification('Backtest completed!', 'success');
        } else {
            showNotification('Backtest failed: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error running backtest:', error);
        showNotification('Failed to run backtest', 'error');
    }
}

function displayBacktestResults(data) {
    const resultsDiv = document.getElementById('backtestResults');
    resultsDiv.style.display = 'block';

    // Update metrics
    document.getElementById('btFinalBalance').textContent = `$${data.metrics.final_balance.toFixed(2)}`;
    document.getElementById('btTotalTrades').textContent = data.metrics.total_trades;
    const winRate = data.metrics.win_rate_percent ?? 0;
    document.getElementById('btWinRate').textContent = `${winRate.toFixed(1)}%`;
    const maxDrawdown = data.metrics.max_drawdown ?? 0;
    const initialBalance = data.metrics.initial_balance || 0;
    const drawdownPct = initialBalance > 0 ? (maxDrawdown / initialBalance) * 100 : 0;
    document.getElementById('btMaxDrawdown').textContent = `${drawdownPct.toFixed(1)}%`;

    // Update chart
    if (backtestChart) {
        backtestChart.destroy();
    }

    const ctx = document.getElementById('backtestChart');
    backtestChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.chart_data.timestamps.map(ts => new Date(ts).toLocaleTimeString()),
            datasets: [
                {
                    label: 'Price',
                    data: data.chart_data.close,
                    borderColor: '#ffffff',
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                    yAxisID: 'y'
                },
                {
                    label: 'EMA Fast',
                    data: data.chart_data.ema_fast,
                    borderColor: '#00d4ff',
                    borderWidth: 2,
                    pointRadius: 0,
                    yAxisID: 'y'
                },
                {
                    label: 'EMA Slow',
                    data: data.chart_data.ema_slow,
                    borderColor: '#7c3aed',
                    borderWidth: 2,
                    pointRadius: 0,
                    yAxisID: 'y'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    position: 'left',
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                },
                x: {
                    ticks: { color: '#94a3b8', maxTicksLimit: 10 },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                }
            }
        }
    });
}

// ============= Trading Controls =============
async function startTrading() {
    try {
        const response = await fetch('/api/trading/start', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            showNotification('Trading started!', 'success');
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
        } else {
            showNotification('Failed to start trading: ' + data.message, 'error');
        }
    } catch (error) {
        console.error('Error starting trading:', error);
        showNotification('Failed to start trading', 'error');
    }
}

async function stopTrading() {
    try {
        const response = await fetch('/api/trading/stop', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            showNotification('Trading stopped!', 'success');
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        } else {
            showNotification('Failed to stop trading: ' + data.message, 'error');
        }
    } catch (error) {
        console.error('Error stopping trading:', error);
        showNotification('Failed to stop trading', 'error');
    }
}

// ============= Status Updates =============
async function updateStatus() {
    try {
        const response = await fetch('/api/trading/status');
        const data = await response.json();

        updateStatusDisplay(data);
        updateTradeHistory();
    } catch (error) {
        console.error('Error updating status:', error);
    }
}

function updateStatusDisplay(data) {
    // Update status indicator
    const statusIndicator = document.getElementById('botStatus');
    const statusText = statusIndicator.querySelector('.status-text');
    statusText.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);

    statusIndicator.className = 'status-indicator ' + data.status;

    // Update stats
    const stats = data.stats || {};
    document.getElementById('totalTrades').textContent = stats.total_trades || 0;
    document.getElementById('dailyTrades').textContent = stats.daily_trades || 0;
    document.getElementById('totalPnl').textContent = `$${(stats.total_pnl || 0).toFixed(2)}`;

    const winRate = stats.total_trades > 0
        ? ((stats.winning_trades / stats.total_trades) * 100).toFixed(1)
        : 0;
    document.getElementById('winRate').textContent = `${winRate}%`;

    // Update trading tab
    document.getElementById('tradingStatus').textContent = data.status;
    document.getElementById('tradingDailyTrades').textContent = stats.daily_trades || 0;
    document.getElementById('tradingDailyPnl').textContent = `$${(stats.daily_pnl || 0).toFixed(2)}`;
    document.getElementById('tradingSymbol').textContent = data.config?.symbol || '-';

    // Update buttons
    if (data.status === 'running') {
        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
    } else {
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
    }
}

function updateStatusFromWebSocket(data) {
    updateStatusDisplay(data);
}

// ============= Trade History =============
async function updateTradeHistory() {
    try {
        const response = await fetch('/api/trades/history?limit=10');
        const data = await response.json();

        if (data.success && data.trades.length > 0) {
            const tbody = document.getElementById('recentTradesBody');
            tbody.innerHTML = '';

            data.trades.reverse().forEach(trade => {
                const row = document.createElement('tr');
                const time = new Date(trade.timestamp).toLocaleTimeString();

                row.innerHTML = `
                    <td>${time}</td>
                    <td><span class="badge ${trade.side.toLowerCase()}">${trade.side}</span></td>
                    <td>$${trade.entry_price.toFixed(2)}</td>
                    <td>$${trade.sl.toFixed(2)}</td>
                    <td>$${trade.tp.toFixed(2)}</td>
                    <td>${trade.status}</td>
                `;
                tbody.appendChild(row);
            });
        }
    } catch (error) {
        console.error('Error updating trade history:', error);
    }
}

// ============= Logs =============
function updateLogs(logs) {
    const container = document.getElementById('logsContainer');
    container.innerHTML = '';

    logs.forEach(log => {
        const entry = document.createElement('div');
        entry.className = 'log-entry';

        const time = new Date(log.timestamp).toLocaleTimeString();
        const level = log.level.toLowerCase();

        entry.innerHTML = `
            <span class="log-time">${time}</span>
            <span class="log-level ${level}">${log.level}</span>
            <span class="log-message">${log.message}</span>
        `;

        container.appendChild(entry);
    });

    // Auto-scroll to bottom
    container.scrollTop = container.scrollHeight;
}

// ============= Event Listeners =============
function setupEventListeners() {
    // Configuration form
    document.getElementById('configForm').addEventListener('submit', (e) => {
        e.preventDefault();
        saveConfiguration();
    });

    // Backtest form
    document.getElementById('backtestForm').addEventListener('submit', (e) => {
        e.preventDefault();
        runBacktest();
    });

    // Trading controls
    document.getElementById('startBtn').addEventListener('click', startTrading);
    document.getElementById('stopBtn').addEventListener('click', stopTrading);
}

// ============= Notifications =============
function showNotification(message, type = 'info') {
    // Simple console notification for now
    // You can implement a toast notification system here
    console.log(`[${type.toUpperCase()}] ${message}`);

    // Optional: Add a visual notification
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 16px 24px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#00d4ff'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        z-index: 1000;
        animation: slideIn 0.3s ease-out;
    `;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}
