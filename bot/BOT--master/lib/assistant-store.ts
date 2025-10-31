import fs from "fs";
import path from "path";

const DB_PATH = path.join(process.cwd(), "server", "assistant_sessions.json");

function ensureDb() {
  const dir = path.dirname(DB_PATH);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  if (!fs.existsSync(DB_PATH))
    fs.writeFileSync(DB_PATH, JSON.stringify({}), "utf-8");
}

export function readAllSessions(): Record<string, any> {
  try {
    ensureDb();
    const raw = fs.readFileSync(DB_PATH, "utf-8");
    return JSON.parse(raw || "{}");
  } catch (e) {
    return {};
  }
}

export function writeAllSessions(obj: Record<string, any>) {
  ensureDb();
  fs.writeFileSync(DB_PATH, JSON.stringify(obj, null, 2), "utf-8");
}

export function getSession(id: string) {
  const db = readAllSessions();
  return db[id] || null;
}

export function saveSession(id: string, data: any) {
  const db = readAllSessions();
  db[id] = { ...(db[id] || {}), ...data };
  writeAllSessions(db);
  return db[id];
}

export function deleteSession(id: string) {
  const db = readAllSessions();
  delete db[id];
  writeAllSessions(db);
}
