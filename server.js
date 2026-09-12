require('dotenv').config();

const express = require('express');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const bcrypt = require('bcryptjs');
const { GoogleGenAI } = require('@google/genai');

const app = express();
const PORT = process.env.PORT || 3000;
const MODEL = 'gemini-3.6-flash';

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

if (!GEMINI_API_KEY || GEMINI_API_KEY.trim() === '' || GEMINI_API_KEY === 'YOUR_API_KEY_HERE') {
    console.error('\n❌ ERROR: GEMINI_API_KEY is missing or not set in .env\n');
    process.exit(1);
}

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

const DATA_DIR = path.join(__dirname, 'data');
const DB_FILE = path.join(DATA_DIR, 'db.json');

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
if (!fs.existsSync(DB_FILE)) {
    fs.writeFileSync(DB_FILE, JSON.stringify({ users: {} }, null, 2));
}

function readDB() {
    return JSON.parse(fs.readFileSync(DB_FILE, 'utf-8'));
}

function writeDB(db) {
    fs.writeFileSync(DB_FILE, JSON.stringify(db, null, 2));
}

const sessions = new Map();

function requireAuth(req, res, next) {
    const authHeader = req.headers['authorization'] || '';
    const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : (req.body && req.body.token);
    const username = sessions.get(token);
    if (!username) {
        return res.status(401).json({ error: 'Not logged in. Please log in again.' });
    }
    req.username = username;
    req.token = token;
    next();
}

app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

app.post('/api/signup', async (req, res) => {
    try {
        const { username, password } = req.body || {};
        if (!username || !password || username.trim() === '' || password.length < 4) {
            return res.status(400).json({ error: 'Username required, password must be at least 4 characters.' });
        }

        const db = readDB();
        if (db.users[username]) {
            return res.status(409).json({ error: 'Username already taken.' });
        }

        const passwordHash = await bcrypt.hash(password, 10);
        db.users[username] = {
            passwordHash,
            lastInteractionId: null,
            messages: [],
        };
        writeDB(db);

        const token = crypto.randomUUID();
        sessions.set(token, username);
        return res.json({ token, username });
    } catch (err) {
        console.error('Signup error:', err);
        return res.status(500).json({ error: 'Could not create account. Please try again.' });
    }
});

app.post('/api/login', async (req, res) => {
    try {
        const { username, password } = req.body || {};
        if (!username || !password) {
            return res.status(400).json({ error: 'Username and password are required.' });
        }

        const db = readDB();
        const user = db.users[username];
        if (!user) {
            return res.status(401).json({ error: 'Invalid username or password.' });
        }

        const match = await bcrypt.compare(password, user.passwordHash);
        if (!match) {
            return res.status(401).json({ error: 'Invalid username or password.' });
        }

        const token = crypto.randomUUID();
        sessions.set(token, username);
        return res.json({ token, username });
    } catch (err) {
        console.error('Login error:', err);
        return res.status(500).json({ error: 'Could not log in. Please try again.' });
    }
});

app.post('/api/logout', requireAuth, (req, res) => {
    sessions.delete(req.token);
    return res.json({ ok: true });
});

app.get('/api/history', requireAuth, (req, res) => {
    const db = readDB();
    const user = db.users[req.username];
    return res.json({ history: (user && user.messages) || [] });
});

app.post('/api/chat', requireAuth, async (req, res) => {
    try {
        const { message } = req.body || {};
        const username = req.username;

        if (!message || typeof message !== 'string' || message.trim() === '') {
            return res.status(400).json({ error: 'Message cannot be empty.' });
        }
        if (message.length > 8000) {
            return res.status(400).json({ error: 'Message is too long (max 8000 characters).' });
        }

        const db = readDB();
        const user = db.users[username];
        if (!user) {
            return res.status(401).json({ error: 'User not found. Please log in again.' });
        }

        const previousInteractionId = user.lastInteractionId || undefined;

        const requestPayload = {
            model: MODEL,
            input: message,
            system_instruction:
                'You are a helpful, friendly AI assistant in a web chat app. Keep answers clear and reasonably concise. ' +
                'If the user ever asks your name, who you are, or what you are called (in any language or phrasing), ' +
                'you must always answer that your name is Sunil Parashar. Never say you are Gemini, an AI, or made by Google.',
        };
        if (previousInteractionId) {
            requestPayload.previous_interaction_id = previousInteractionId;
        }

        const interaction = await ai.interactions.create(requestPayload);
        const reply = interaction && interaction.output_text ? interaction.output_text.trim() : '';

        if (!reply) {
            return res.status(502).json({ error: 'The AI returned an empty response. Please try again.' });
        }

        user.lastInteractionId = interaction.id || user.lastInteractionId;
        user.messages.push({ role: 'user', text: message });
        user.messages.push({ role: 'ai', text: reply });
        writeDB(db);

        return res.json({ reply });
    } catch (err) {
        return handleGeminiError(err, res);
    }
});

function handleGeminiError(err, res) {
    console.error('Gemini API error:', err && err.message ? err.message : err);

    const status = err && (err.status || err.httpStatus || (err.response && err.response.status));
    const rawMessage = err && err.message ? err.message.toLowerCase() : '';

    if (status === 401 || status === 403 || rawMessage.includes('api key') || rawMessage.includes('permission')) {
        return res.status(401).json({ error: 'Invalid or unauthorized Gemini API key. Check your .env file.' });
    }
    if (status === 404 || (rawMessage.includes('model') && rawMessage.includes('not'))) {
        return res.status(404).json({ error: 'The model "' + MODEL + '" is unavailable right now.' });
    }
    if (status === 429 || rawMessage.includes('quota') || rawMessage.includes('rate limit')) {
        return res.status(429).json({ error: 'Rate limit or quota exceeded on the Gemini API. Please wait and try again.' });
    }
    if (
        rawMessage.includes('fetch failed') ||
        rawMessage.includes('econnrefused') ||
        rawMessage.includes('enotfound') ||
        rawMessage.includes('timeout') ||
        err.code === 'ECONNREFUSED' ||
        err.code === 'ENOTFOUND'
    ) {
        return res.status(503).json({ error: 'Network error while contacting the Gemini API.' });
    }
    if (status === 400) {
        return res.status(400).json({ error: 'The request to Gemini was invalid. ' + (err.message || '') });
    }

    return res.status(500).json({ error: 'Something went wrong on the server while talking to Gemini.' });
}

app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
    console.log('\n✅ My AI Chat server running at http://localhost:' + PORT);
    console.log('   Using model: ' + MODEL + '\n');
});