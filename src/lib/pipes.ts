import { supabase } from "./supabaseClient";
import type { Pipe } from "../types/pipe";

type PipeRow = {
  id: string;
  name: string;
  description: string | null;
  type: Pipe["type"];
  status: Pipe["status"];
  is_template: boolean;
  created_at: string;
  updated_at: string;
};

export async function getPipes(): Promise<Pipe[]> {
  const { data, error } = await supabase
    .from("pipes")
    .select(
      "id, name, description, type, status, is_template, created_at, updated_at",
    )
    .order("created_at", { ascending: true });

  if (error) {
    throw error;
  }

  return ((data ?? []) as PipeRow[]).map((pipe) => ({
    id: pipe.id,
    name: pipe.name,
    description: pipe.description ?? "",
    type: pipe.type,
    status: pipe.status,
    isTemplate: pipe.is_template,
    createdAt: pipe.created_at,
    updatedAt: pipe.updated_at,
  }));
}

export async function getPipeById(pipeId: string): Promise<Pipe | null> {
  const { data, error } = await supabase
    .from("pipes")
    .select(
      "id, name, description, type, status, is_template, created_at, updated_at",
    )
    .eq("id", pipeId)
    .maybeSingle();

  if (error) {
    throw error;
  }

  if (!data) return null;

  const pipe = data as PipeRow;

  return {
    id: pipe.id,
    name: pipe.name,
    description: pipe.description ?? "",
    type: pipe.type,
    status: pipe.status,
    isTemplate: pipe.is_template,
    createdAt: pipe.created_at,
    updatedAt: pipe.updated_at,
  };
}