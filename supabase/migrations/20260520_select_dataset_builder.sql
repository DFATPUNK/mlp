create table if not exists provider_connections (
  id text primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
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
  user_id uuid not null references auth.users(id) on delete cascade,
  provider text not null,
  connection_id text references provider_connections(id) on delete set null,
  external_id text not null,
  external_name text not null,
  external_url text,
  source_config jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists dataset_profiles (
  id uuid primary key default gen_random_uuid(),
  pipe_id uuid not null references pipes(id) on delete cascade,
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
  pipe_id uuid not null references pipes(id) on delete cascade,
  step_key text not null,
  artifact_id uuid references artifacts(id) on delete set null,
  output jsonb not null,
  status text not null,
  created_at timestamptz not null default now(),
  unique(pipe_id, step_key)
);

alter table provider_connections enable row level security;
alter table dataset_sources enable row level security;
alter table dataset_profiles enable row level security;
alter table pipe_step_outputs enable row level security;

drop policy if exists "provider_connections_owner_rw" on provider_connections;
create policy "provider_connections_owner_rw" on provider_connections
for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists "dataset_sources_owner_rw" on dataset_sources;
create policy "dataset_sources_owner_rw" on dataset_sources
for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists "dataset_profiles_owner_rw" on dataset_profiles;
create policy "dataset_profiles_owner_rw" on dataset_profiles
for all using (exists (select 1 from pipes p where p.id = dataset_profiles.pipe_id and p.owner_id = auth.uid()))
with check (exists (select 1 from pipes p where p.id = dataset_profiles.pipe_id and p.owner_id = auth.uid()));

drop policy if exists "pipe_step_outputs_owner_rw" on pipe_step_outputs;
create policy "pipe_step_outputs_owner_rw" on pipe_step_outputs
for all using (exists (select 1 from pipes p where p.id = pipe_step_outputs.pipe_id and p.owner_id = auth.uid()))
with check (exists (select 1 from pipes p where p.id = pipe_step_outputs.pipe_id and p.owner_id = auth.uid()));
