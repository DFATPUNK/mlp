create table if not exists public.pipe_publications (
  pipe_id uuid primary key references public.pipes(id) on delete cascade,
  public_id uuid not null unique default gen_random_uuid(),
  api_key_hash text not null,
  api_key_prefix text not null,
  active_version integer not null default 0,
  is_live boolean not null default false,
  active_deployment_id uuid null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  published_at timestamptz null,
  unpublished_at timestamptz null,
  last_key_rotated_at timestamptz null
);

create table if not exists public.pipe_deployments (
  id uuid primary key default gen_random_uuid(),
  pipe_id uuid not null references public.pipes(id) on delete cascade,
  version integer not null,
  status text not null check (status in ('active', 'superseded', 'unpublished')),
  trained_models_artifact_id uuid not null references public.artifacts(id),
  review_results_artifact_id uuid not null references public.artifacts(id),
  test_prediction_artifact_id uuid null references public.artifacts(id),
  input_schema jsonb not null,
  output_schema jsonb not null,
  model_snapshot jsonb not null,
  published_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  unique (pipe_id, version)
);

alter table public.pipe_publications
  add constraint pipe_publications_active_deployment_id_fkey
  foreign key (active_deployment_id)
  references public.pipe_deployments(id);

create index if not exists pipe_publications_public_id_idx on public.pipe_publications(public_id);
create index if not exists pipe_publications_live_idx on public.pipe_publications(public_id, is_live);
create index if not exists pipe_deployments_pipe_status_idx on public.pipe_deployments(pipe_id, status);

alter table public.pipe_publications enable row level security;
alter table public.pipe_deployments enable row level security;

drop policy if exists "pipe_publications_owner_select" on public.pipe_publications;
create policy "pipe_publications_owner_select"
on public.pipe_publications
for select
to authenticated
using (exists (select 1 from public.pipes p where p.id = pipe_publications.pipe_id and p.owner_id = auth.uid()));

drop policy if exists "pipe_deployments_owner_select" on public.pipe_deployments;
create policy "pipe_deployments_owner_select"
on public.pipe_deployments
for select
to authenticated
using (exists (select 1 from public.pipes p where p.id = pipe_deployments.pipe_id and p.owner_id = auth.uid()));
