create table if not exists provider_connections (
  id text primary key,
  user_id uuid not null,
  provider text not null,
  provider_account_id text not null,
  provider_account_label text not null,
  scopes text[] not null default '{}',
  expires_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists dataset_sources (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  provider text not null,
  connection_id text,
  external_id text not null,
  external_name text not null,
  external_url text,
  source_config jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists dataset_profiles (
  id uuid primary key default gen_random_uuid(),
  pipe_id uuid not null,
  dataset_source_id uuid not null references dataset_sources(id) on delete cascade,
  row_count integer not null,
  column_count integer not null,
  columns jsonb not null,
  missing_values jsonb not null,
  eligibility jsonb not null,
  preview jsonb not null,
  created_at timestamptz not null default now()
);

create table if not exists pipe_step_outputs (
  id uuid primary key default gen_random_uuid(),
  pipe_id uuid not null,
  step_key text not null,
  artifact_id uuid,
  output jsonb not null,
  status text not null,
  created_at timestamptz not null default now(),
  unique(pipe_id, step_key)
);
