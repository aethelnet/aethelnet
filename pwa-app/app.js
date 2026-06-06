/**
 * Sovereign Swarm MVP - Core Logic
 * Clean, efficient vanilla JS. No bloat.
 */

class SwarmClient {
    constructor() {
        // We'll target the local ProphitEngine WS port for now (assuming it's running locally).
        // Adapt the port to wherever the engine is exposed (e.g. ws://localhost:8001/ws)
        this.wsUrl = 'ws://localhost:8001/ws'; // Adjust based on the actual backend route
        this.socket = null;
        this.reconnectTimer = null;
        this.reconnectAttempts = 0;
        
        // DOM Elements
        this.elStatus = document.getElementById('ws-status');
        this.elLight = document.getElementById('ws-light');
        this.elLog = document.getElementById('event-log');
        
        // Telemetry DOM
        this.elNodes = document.getElementById('val-nodes');
        this.elBridges = document.getElementById('val-bridges');
        this.elState = document.getElementById('val-state');

        this.init();
    }

    init() {
        this.log('Initializing Sovereign Swarm Client...', 'info');
        this.connect();
        
        document.getElementById('btn-reconnect').addEventListener('click', () => {
            if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                this.log('Force syncing state...', 'info');
                this.socket.send(JSON.stringify({ type: 'sync_request' }));
            } else {
                this.connect();
            }
        });
    }

    connect() {
        if (this.socket && (this.socket.readyState === WebSocket.CONNECTING || this.socket.readyState === WebSocket.OPEN)) {
            return;
        }

        this.updateStatus('Connecting...', 'connecting');
        this.log(`Attempting connection to ${this.wsUrl}`, 'info');

        try {
            this.socket = new WebSocket(this.wsUrl);

            this.socket.onopen = (e) => this.onOpen(e);
            this.socket.onmessage = (e) => this.onMessage(e);
            this.socket.onclose = (e) => this.onClose(e);
            this.socket.onerror = (e) => this.onError(e);
        } catch (error) {
            this.log(`WebSocket instantiation error: ${error.message}`, 'error');
            this.updateStatus('Disconnected', 'disconnected');
        }
    }

    onOpen(event) {
        this.reconnectAttempts = 0;
        this.updateStatus('Connected', 'connected');
        this.log('Connection established with the Engine.', 'success');
        
        // Request initial state if the protocol requires it
        this.socket.send(JSON.stringify({ type: 'hello', client: 'pwa_mvp' }));
    }

    onMessage(event) {
        try {
            const data = JSON.parse(event.data);
            this.handleTelemetry(data);
        } catch (e) {
            // If it's not JSON, just log it as a raw string
            this.log(`RAW: ${event.data}`, 'info');
        }
    }

    onClose(event) {
        this.updateStatus('Disconnected', 'disconnected');
        this.log(`Connection closed. Code: ${event.code}`, 'error');
        this.scheduleReconnect();
    }

    onError(event) {
        this.log('WebSocket Error occurred.', 'error');
    }

    scheduleReconnect() {
        this.reconnectAttempts++;
        // Exponential backoff up to 10 seconds
        const delay = Math.min(1000 * Math.pow(1.5, this.reconnectAttempts), 10000);
        
        this.log(`Reconnecting in ${(delay/1000).toFixed(1)}s...`, 'info');
        
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = setTimeout(() => {
            this.connect();
        }, delay);
    }

    handleTelemetry(data) {
        // Here we parse whatever the ProphitEngine/Aethelnet sends.
        // This is a placeholder structure. We will adapt this to the actual payload later.
        
        if (data.type === 'telemetry' || data.nodes !== undefined) {
            if (data.nodes !== undefined) this.elNodes.textContent = data.nodes;
            if (data.bridges !== undefined) this.elBridges.textContent = data.bridges;
            if (data.state !== undefined) this.elState.textContent = data.state;
            
            // Log less frequently to avoid spam, or only log significant events
        } else if (data.type === 'event' || data.message) {
            this.log(data.message || JSON.stringify(data), 'info');
        }
    }

    updateStatus(text, stateClass) {
        this.elStatus.textContent = text;
        this.elLight.className = `indicator-light ${stateClass}`;
    }

    log(message, type = 'info') {
        const time = new Date().toLocaleTimeString('en-US', { hour12: false });
        const el = document.createElement('div');
        el.className = 'log-entry';
        
        el.innerHTML = `<span class="log-time">[${time}]</span><span class="log-msg ${type}">${message}</span>`;
        
        this.elLog.appendChild(el);
        
        // Auto-scroll to bottom
        this.elLog.scrollTop = this.elLog.scrollHeight;
        
        // Keep log size manageable
        while (this.elLog.children.length > 50) {
            this.elLog.removeChild(this.elLog.firstChild);
        }
    }
}

// Boot the client when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.swarmClient = new SwarmClient();
});
