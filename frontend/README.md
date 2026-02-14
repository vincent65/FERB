# FERB Chat Frontend

Whitespace-style AI chatbot for kernel optimization and faster ML training.

## Run everything (API + frontend)

From the **project root** (FERB):

```bash
cd api && pip install -r requirements.txt && uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://127.0.0.1:8000** in your browser. The same server serves the UI and the `/chat` API.

## Run frontend only (e.g. static server)

If the API runs elsewhere, open `index.html` or serve `frontend/` with any static server and set the API base:

```html
<script>window.FERB_API_BASE = "http://localhost:8000";</script>
<script src="app.js"></script>
```

Or run from `frontend/`:

```bash
npx serve -p 3000
```

Then set `FERB_API_BASE` to your API URL (e.g. `http://127.0.0.1:8000`) if the API is on a different port.
