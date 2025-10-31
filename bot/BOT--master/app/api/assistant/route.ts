import { NextResponse } from "next/server";
import { v4 as uuidv4 } from "uuid";
import { getSession, saveSession, deleteSession } from "@/lib/assistant-store";

export async function POST(req: Request) {
  const url = new URL(req.url);
  const id = url.searchParams.get("id");
  if (!id) {
    // create new session
    const nid = uuidv4();
    saveSession(nid, { createdAt: Date.now(), messages: [] });
    return NextResponse.json({ id: nid });
  }

  // update existing session metadata (e.g., name) or append messages
  try {
    const body = await req.json();
    const existing = getSession(id) || { messages: [] };
    if (body.name) existing.name = body.name;
    if (body.messages)
      existing.messages = [...(existing.messages || []), ...body.messages];
    saveSession(id, existing);
    return NextResponse.json({ ok: true });
  } catch (e) {
    return NextResponse.json({ error: "bad body" }, { status: 400 });
  }
}

export async function GET(req: Request) {
  try {
    const url = new URL(req.url);
    const id = url.searchParams.get("id");
    if (!id)
      return NextResponse.json({ error: "id required" }, { status: 400 });
    const session = getSession(id);
    if (!session)
      return NextResponse.json({ error: "not found" }, { status: 404 });
    return NextResponse.json({ session });
  } catch (e) {
    return NextResponse.json({ error: "bad request" }, { status: 400 });
  }
}

export async function DELETE(req: Request) {
  try {
    const url = new URL(req.url);
    const id = url.searchParams.get("id");
    if (!id)
      return NextResponse.json({ error: "id required" }, { status: 400 });
    deleteSession(id);
    return NextResponse.json({ ok: true });
  } catch (e) {
    return NextResponse.json({ error: "bad request" }, { status: 400 });
  }
}
