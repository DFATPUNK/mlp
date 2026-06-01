-- Link dataset sources directly to their owning pipes so pipe deletion can
-- cascade through dataset sources and their dataset profiles.
alter table public.dataset_sources
add column if not exists pipe_id uuid;

update public.dataset_sources ds
set pipe_id = dp.pipe_id
from public.dataset_profiles dp
where dp.dataset_source_id = ds.id
  and ds.pipe_id is null;

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'dataset_sources_pipe_id_delete_cascade_fkey'
      and conrelid = 'public.dataset_sources'::regclass
  ) then
    alter table public.dataset_sources
      add constraint dataset_sources_pipe_id_delete_cascade_fkey
      foreign key (pipe_id)
      references public.pipes(id)
      on delete cascade
      not valid;
  end if;
end $$;

-- Keep existing artifacts data intact while enforcing ownership cleanup for
-- future and currently valid rows. NOT VALID avoids blocking this migration if
-- production contains a historical artifact whose pipe was already removed.
do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'artifacts_pipe_id_delete_cascade_fkey'
      and conrelid = 'public.artifacts'::regclass
  ) then
    alter table public.artifacts
      add constraint artifacts_pipe_id_delete_cascade_fkey
      foreign key (pipe_id)
      references public.pipes(id)
      on delete cascade
      not valid;
  end if;
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

-- Allow the frontend's best-effort rollback of partially-created artifacts.
-- Pipe deletion also cascades artifacts through the FK above.
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

-- Optional one-time cleanup for historical orphan sources. Review before use:
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
