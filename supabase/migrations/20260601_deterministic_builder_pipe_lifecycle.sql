-- Deterministic builder lifecycle:
-- every new dataset source is linked to an existing draft pipe, and deleting
-- that pipe cascades to its source rows and downstream profiles.
alter table public.dataset_sources
add column if not exists pipe_id uuid;

update public.dataset_sources ds
set pipe_id = dp.pipe_id
from public.dataset_profiles dp
where dp.dataset_source_id = ds.id
  and ds.pipe_id is null;

-- Replace any previous dataset_sources.pipe_id FK with an explicit cascade FK.
do $$
declare
  constraint_name text;
begin
  for constraint_name in
    select c.conname
    from pg_constraint c
    join pg_attribute a
      on a.attrelid = c.conrelid
     and a.attnum = any (c.conkey)
    where c.contype = 'f'
      and c.conrelid = 'public.dataset_sources'::regclass
      and c.confrelid = 'public.pipes'::regclass
      and a.attname = 'pipe_id'
  loop
    execute format('alter table public.dataset_sources drop constraint %I', constraint_name);
  end loop;

  alter table public.dataset_sources
    add constraint dataset_sources_pipe_id_fkey
    foreign key (pipe_id)
    references public.pipes(id)
    on delete cascade
    not valid;
end $$;

-- Artifacts should follow their pipe lifecycle too. NOT VALID preserves
-- historical data while enforcing the FK for future inserts and deletes.
do $$
declare
  constraint_name text;
begin
  for constraint_name in
    select c.conname
    from pg_constraint c
    join pg_attribute a
      on a.attrelid = c.conrelid
     and a.attnum = any (c.conkey)
    where c.contype = 'f'
      and c.conrelid = 'public.artifacts'::regclass
      and c.confrelid = 'public.pipes'::regclass
      and a.attname = 'pipe_id'
  loop
    execute format('alter table public.artifacts drop constraint %I', constraint_name);
  end loop;

  alter table public.artifacts
    add constraint artifacts_pipe_id_fkey
    foreign key (pipe_id)
    references public.pipes(id)
    on delete cascade
    not valid;
end $$;

alter table public.dataset_sources enable row level security;

drop policy if exists "dataset_sources_owner_rw" on public.dataset_sources;
create policy "dataset_sources_owner_rw"
on public.dataset_sources
for all
to authenticated
using (
  user_id = auth.uid()
  and (
    pipe_id is null
    or exists (
      select 1
      from public.pipes p
      where p.id = dataset_sources.pipe_id
        and p.owner_id = auth.uid()
    )
  )
)
with check (
  user_id = auth.uid()
  and (
    pipe_id is null
    or exists (
      select 1
      from public.pipes p
      where p.id = dataset_sources.pipe_id
        and p.owner_id = auth.uid()
    )
  )
);

-- Allow best-effort rollback of a dataset artifact if a later frontend write
-- fails. Pipe deletion itself cascades artifacts through the FK above.
drop policy if exists "artifacts_owner_delete" on public.artifacts;
create policy "artifacts_owner_delete"
on public.artifacts
for delete
to authenticated
using (
  exists (
    select 1
    from public.pipes p
    where p.id = artifacts.pipe_id
      and p.owner_id = auth.uid()
  )
);

-- Manual cleanup for historical dataset_sources that were never linked.
-- Review these rows before running this destructive cleanup:
-- delete from public.dataset_sources ds
-- where ds.pipe_id is null
--   and not exists (
--     select 1
--     from public.dataset_profiles dp
--     where dp.dataset_source_id = ds.id
--   );

-- Diagnostic: historical orphan dataset sources.
-- select *
-- from public.dataset_sources ds
-- where ds.pipe_id is null
--   and not exists (
--     select 1
--     from public.dataset_profiles dp
--     where dp.dataset_source_id = ds.id
--   );

-- Diagnostic: dataset sources and profiles per pipe.
-- select
--   p.id as pipe_id,
--   p.type,
--   p.status,
--   ds.id as dataset_source_id,
--   dp.id as dataset_profile_id
-- from public.pipes p
-- left join public.dataset_sources ds on ds.pipe_id = p.id
-- left join public.dataset_profiles dp on dp.dataset_source_id = ds.id
-- order by p.created_at desc;
