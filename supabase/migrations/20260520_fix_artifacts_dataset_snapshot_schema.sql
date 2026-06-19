alter table public.artifacts
add column if not exists kind text;

alter table public.artifacts
add column if not exists content jsonb;

alter table public.artifacts
add column if not exists metadata jsonb;

alter table public.artifacts
add column if not exists name text;

alter table public.artifacts
add column if not exists artifact_type text;

alter table public.artifacts
alter column artifact_type set default 'dataset_snapshot';

alter table public.artifacts
alter column metadata set default '{}'::jsonb;

alter table public.artifacts
alter column name set default 'artifact';

alter table public.artifacts
enable row level security;

drop policy if exists "artifacts_owner_insert" on public.artifacts;
create policy "artifacts_owner_insert"
on public.artifacts
for insert
to authenticated
with check (
  exists (
    select 1
    from public.pipes p
    where p.id = artifacts.pipe_id
      and p.owner_id = auth.uid()
  )
);

drop policy if exists "artifacts_owner_select" on public.artifacts;
create policy "artifacts_owner_select"
on public.artifacts
for select
to authenticated
using (
  exists (
    select 1
    from public.pipes p
    where p.id = artifacts.pipe_id
      and (
        p.owner_id = auth.uid()
        or p.is_template = true
      )
  )
);
