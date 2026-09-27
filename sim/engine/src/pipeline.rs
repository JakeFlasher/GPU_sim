//! Deterministic dependency/resource scheduler, independent of any vendor ISA.
//! Each group issues in order, one instruction per cycle. Resource service is
//! pipelined: initiation interval, transfer width, latency and in-flight depth
//! are separate. Completion events free resource and residency capacity.

use crate::{EventQueue, TargetId};

#[derive(Debug, Clone)]
pub struct Resource {
    pub width: u32,
    pub latency: u64,
    pub initiation_interval: u64,
    pub max_in_flight: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct Task {
    pub resource: Option<usize>,
    /// Optional alternatives; empty preserves the legacy single binding.
    pub resource_options: Vec<usize>,
    pub work: u64,
    /// Completion dependencies on earlier tasks in this group.
    pub dependencies: Vec<usize>,
    /// Completion dependencies on tasks in another resident group. This is
    /// the generic event-engine representation of a bounded group barrier.
    pub cross_dependencies: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval {
    pub start: u64,
    pub complete: u64,
}

#[derive(Debug)]
pub struct Schedule {
    pub groups: Vec<Vec<Interval>>,
    /// Concrete resource selected for each task at issue time.
    pub selected_resources: Vec<Vec<Option<usize>>>,
    pub admissions: Vec<u64>,
    pub retirements: Vec<u64>,
    pub final_cycle: u64,
    pub completion_events: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("invalid pipeline input: {0}")]
    Invalid(&'static str),
    #[error("pipeline cycle arithmetic overflow")]
    Overflow,
    #[error("pipeline deadlock: blocked (group, task) heads {blocked_heads:?}")]
    Deadlock { blocked_heads: Vec<(usize, usize)> },
    #[error(transparent)]
    Engine(#[from] crate::EngineError),
}

pub fn schedule(
    resources: &[Resource],
    groups: &[Vec<Task>],
    max_resident: u32,
) -> Result<Schedule, PipelineError> {
    if groups.is_empty() || groups.len() > 64 || max_resident == 0 || max_resident > 64 {
        return Err(PipelineError::Invalid(
            "require 1..=64 groups and resident capacity",
        ));
    }
    for r in resources {
        if r.width == 0
            || r.latency == 0
            || r.initiation_interval == 0
            || r.max_in_flight == Some(0)
        {
            return Err(PipelineError::Invalid(
                "resource width/latency/interval/capacity must be positive",
            ));
        }
    }
    for (g, group) in groups.iter().enumerate() {
        if group.is_empty() || group.len() > 4096 {
            return Err(PipelineError::Invalid("require 1..=4096 tasks per group"));
        }
        for (i, task) in group.iter().enumerate() {
            if task.resource.is_some_and(|r| r >= resources.len())
                || task.resource_options.iter().any(|r| *r >= resources.len())
                || (task.resource.is_some() && !task.resource_options.is_empty())
                || task.resource_options.windows(2).any(|w| w[0] >= w[1])
                || task.dependencies.iter().any(|d| *d >= i)
                || task.cross_dependencies.iter().any(|(owner, task)| {
                    groups.get(*owner).is_none_or(|group| *task >= group.len())
                        || (*owner == g && *task >= i)
                })
                || ((task.resource.is_some() || !task.resource_options.is_empty())
                    && task.work == 0)
            {
                return Err(PipelineError::Invalid(
                    "bad resource, work or dependency index",
                ));
            }
        }
    }
    let count = groups.len();
    let mut result = Schedule {
        groups: vec![Vec::new(); count],
        selected_resources: vec![Vec::new(); count],
        admissions: vec![0; count],
        retirements: vec![0; count],
        final_cycle: 0,
        completion_events: 0,
    };
    let mut active = vec![false; count];
    let mut outstanding = vec![0usize; count];
    let mut group_next = vec![0u64; count];
    let mut resource_next = vec![0u64; resources.len()];
    let mut pending = vec![Vec::<u64>::new(); resources.len()];
    let mut queue = EventQueue::<(usize, Option<usize>)>::new();
    let mut next_group = 0;
    let mut retired = 0;
    let mut now = 0;
    let mut round_robin = 0;
    while retired < count {
        while next_group < count && active.iter().filter(|v| **v).count() < max_resident as usize {
            active[next_group] = true;
            result.admissions[next_group] = now;
            group_next[next_group] = now;
            next_group += 1;
        }
        // Choose earliest ready head; rotate equal-time contention fairly.
        let mut choice: Option<(u64, usize, Option<usize>)> = None;
        for offset in 0..count {
            let g = (round_robin + offset) % count;
            let i = result.groups[g].len();
            if !active[g] || i == groups[g].len() {
                continue;
            }
            let task = &groups[g][i];
            let mut start = now.max(group_next[g]);
            for dep in &task.dependencies {
                start = start.max(result.groups[g][*dep].complete);
            }
            let mut cross_ready = true;
            for (dep_group, dep_task) in &task.cross_dependencies {
                if *dep_task >= result.groups[*dep_group].len() {
                    cross_ready = false;
                    break;
                }
                start = start.max(result.groups[*dep_group][*dep_task].complete);
            }
            if !cross_ready {
                continue;
            }
            let candidates = if task.resource_options.is_empty() {
                task.resource.into_iter().collect::<Vec<_>>()
            } else {
                task.resource_options.clone()
            };
            let mut selected = None;
            let mut selected_start = start;
            let mut selected_complete = u64::MAX;
            let mut saw_overflow = false;
            for r in candidates {
                let mut candidate_start = start.max(resource_next[r]);
                if resources[r]
                    .max_in_flight
                    .is_some_and(|cap| pending[r].len() >= cap as usize)
                {
                    candidate_start = candidate_start.max(
                        *pending[r]
                            .iter()
                            .min()
                            .ok_or(PipelineError::Invalid("full empty queue"))?,
                    );
                }
                let cfg = &resources[r];
                let span = task.work.div_ceil(u64::from(cfg.width));
                let Some(service) = span.checked_mul(cfg.initiation_interval) else {
                    saw_overflow = true;
                    continue;
                };
                // A one-beat operation can complete well before the end of
                // a long initiation interval. The reservation endpoint must
                // be representable too, not just its completion timestamp.
                // Otherwise a seemingly faster but unrepresentable route
                // would hide a viable alternative and fail only at issue.
                if candidate_start.checked_add(service).is_none() {
                    saw_overflow = true;
                    continue;
                }
                let Some(last_beat_end) = service
                    .checked_sub(cfg.initiation_interval)
                    .and_then(|v| v.checked_add(1))
                else {
                    saw_overflow = true;
                    continue;
                };
                let Some(candidate_complete) = candidate_start
                    .checked_add(last_beat_end)
                    .and_then(|v| v.checked_add(cfg.latency))
                else {
                    saw_overflow = true;
                    continue;
                };
                if selected.is_none_or(|best| {
                    (candidate_complete, candidate_start, r)
                        < (selected_complete, selected_start, best)
                }) {
                    selected = Some(r);
                    selected_start = candidate_start;
                    selected_complete = candidate_complete;
                }
            }
            if selected.is_none() && saw_overflow {
                return Err(PipelineError::Overflow);
            }
            start = selected_start;
            if choice.is_none_or(|(best, _, _)| start < best) {
                choice = Some((start, g, selected));
            }
        }
        // Completion before same-time issue: release capacity, then arbitrate.
        if queue
            .peek_time()
            .is_some_and(|t| choice.is_none_or(|(start, _, _)| t <= start))
        {
            let event = queue
                .pop()?
                .ok_or(PipelineError::Invalid("missing completion"))?;
            now = now.max(event.time);
            let (g, resource) = event.kind;
            outstanding[g] -= 1;
            if let Some(r) = resource {
                let ix = pending[r]
                    .iter()
                    .position(|t| *t == event.time)
                    .ok_or(PipelineError::Invalid("completion lost resource owner"))?;
                pending[r].swap_remove(ix);
            }
            if outstanding[g] == 0 && result.groups[g].len() == groups[g].len() {
                active[g] = false;
                result.retirements[g] = now;
                retired += 1;
            }
            continue;
        }
        let (start, g, selected_resource) = choice.ok_or_else(|| PipelineError::Deadlock {
            blocked_heads: active
                .iter()
                .enumerate()
                .filter(|(_, active)| **active)
                .map(|(g, _)| (g, result.groups[g].len()))
                .collect(),
        })?;
        now = start;
        let task = &groups[g][result.groups[g].len()];
        result.selected_resources[g].push(selected_resource);
        let complete = if let Some(r) = selected_resource {
            let cfg = &resources[r];
            let span = task.work.div_ceil(u64::from(cfg.width));
            let service = span
                .checked_mul(cfg.initiation_interval)
                .ok_or(PipelineError::Overflow)?;
            resource_next[r] = start.checked_add(service).ok_or(PipelineError::Overflow)?;
            // The last beat starts (span-1)*II after the first; latency
            // begins after that one-cycle beat, not after `span` cycles
            // when II > 1. Otherwise a transfer could complete before its
            // last beat has even been initiated.
            let last_beat_end = service
                .checked_sub(cfg.initiation_interval)
                .and_then(|t| t.checked_add(1))
                .ok_or(PipelineError::Overflow)?;
            let complete = start
                .checked_add(last_beat_end)
                .and_then(|t| t.checked_add(cfg.latency))
                .ok_or(PipelineError::Overflow)?;
            pending[r].push(complete);
            complete
        } else {
            start
        };
        group_next[g] = start.checked_add(1).ok_or(PipelineError::Overflow)?;
        result.groups[g].push(Interval { start, complete });
        outstanding[g] += 1;
        queue.schedule_at(complete, TargetId(g as u32), (g, selected_resource))?;
        round_robin = (g + 1) % count;
    }
    result.final_cycle = now;
    result.completion_events = queue.stats().events_popped;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_resources_overlap_but_dependents_wait() {
        let result = schedule(
            &[
                Resource {
                    width: 1,
                    latency: 1,
                    initiation_interval: 1,
                    max_in_flight: None,
                },
                Resource {
                    width: 8,
                    latency: 20,
                    initiation_interval: 1,
                    max_in_flight: None,
                },
            ],
            &[vec![
                Task {
                    resource: Some(0),
                    resource_options: vec![],
                    work: 1,
                    dependencies: vec![],
                    cross_dependencies: vec![],
                },
                Task {
                    resource: Some(1),
                    resource_options: vec![],
                    work: 8,
                    dependencies: vec![],
                    cross_dependencies: vec![],
                },
                Task {
                    resource: Some(0),
                    resource_options: vec![],
                    work: 1,
                    dependencies: vec![1],
                    cross_dependencies: vec![],
                },
            ]],
            1,
        )
        .expect("independent-resource schedule should succeed");
        assert_eq!(result.groups[0][1].start, 1);
        assert_eq!(result.groups[0][2].start, 22);
        assert_eq!(result.final_cycle, 24);
    }

    #[test]
    fn resident_capacity_admits_second_group_after_retirement() {
        let result = schedule(
            &[Resource {
                width: 1,
                latency: 4,
                initiation_interval: 1,
                max_in_flight: None,
            }],
            &[
                vec![Task {
                    resource: Some(0),
                    resource_options: vec![],
                    work: 1,
                    dependencies: vec![],
                    cross_dependencies: vec![],
                }],
                vec![Task {
                    resource: Some(0),
                    resource_options: vec![],
                    work: 1,
                    dependencies: vec![],
                    cross_dependencies: vec![],
                }],
            ],
            1,
        )
        .expect("resident-capacity schedule should succeed");
        assert_eq!(result.admissions, vec![0, 5]);
        assert_eq!(result.groups[1][0].start, 5);
    }

    #[test]
    fn in_flight_capacity_delays_even_when_initiation_is_available() {
        let result = schedule(
            &[Resource {
                width: 1,
                latency: 5,
                initiation_interval: 1,
                max_in_flight: Some(1),
            }],
            &[
                vec![Task {
                    resource: Some(0),
                    resource_options: vec![],
                    work: 1,
                    dependencies: vec![],
                    cross_dependencies: vec![],
                }],
                vec![Task {
                    resource: Some(0),
                    resource_options: vec![],
                    work: 1,
                    dependencies: vec![],
                    cross_dependencies: vec![],
                }],
            ],
            2,
        )
        .expect("in-flight-capacity schedule should succeed");
        assert_eq!(result.groups[0][0].start, 0);
        assert_eq!(result.groups[1][0].start, 6);
    }

    #[test]
    fn checked_completion_overflow_is_rejected() {
        let error = schedule(
            &[Resource {
                width: 1,
                latency: u64::MAX,
                initiation_interval: 1,
                max_in_flight: None,
            }],
            &[vec![Task {
                resource: Some(0),
                resource_options: vec![],
                work: 1,
                dependencies: vec![],
                cross_dependencies: vec![],
            }]],
            1,
        )
        .expect_err("latency addition must be checked");
        assert!(matches!(error, PipelineError::Overflow));
    }

    #[test]
    fn malformed_dependency_and_resource_indices_fail_closed() {
        assert!(matches!(
            schedule(
                &[Resource {
                    width: 1,
                    latency: 1,
                    initiation_interval: 1,
                    max_in_flight: None
                }],
                &[vec![Task {
                    resource: Some(1),
                    resource_options: vec![],
                    work: 1,
                    dependencies: vec![],
                    cross_dependencies: vec![],
                }]],
                1,
            ),
            Err(PipelineError::Invalid(_))
        ));
        assert!(matches!(
            schedule(
                &[Resource {
                    width: 1,
                    latency: 1,
                    initiation_interval: 1,
                    max_in_flight: None
                }],
                &[vec![Task {
                    resource: Some(0),
                    resource_options: vec![],
                    work: 1,
                    dependencies: vec![1],
                    cross_dependencies: vec![],
                }]],
                1,
            ),
            Err(PipelineError::Invalid(_))
        ));
    }

    #[test]
    fn alternative_resources_choose_earliest_completion_and_report_choice() {
        let result = schedule(
            &[
                Resource {
                    width: 1,
                    latency: 8,
                    initiation_interval: 4,
                    max_in_flight: Some(1),
                },
                Resource {
                    width: 1,
                    latency: 1,
                    initiation_interval: 1,
                    max_in_flight: Some(1),
                },
            ],
            &[vec![
                Task {
                    resource: Some(0),
                    resource_options: vec![],
                    work: 1,
                    dependencies: vec![],
                    cross_dependencies: vec![],
                },
                Task {
                    resource: None,
                    resource_options: vec![0, 1],
                    work: 1,
                    dependencies: vec![],
                    cross_dependencies: vec![],
                },
                Task {
                    resource: None,
                    resource_options: vec![0, 1],
                    work: 1,
                    dependencies: vec![],
                    cross_dependencies: vec![],
                },
            ]],
            1,
        )
        .expect("alternative-resource schedule should succeed");
        assert_eq!(
            result.selected_resources[0],
            vec![Some(0), Some(1), Some(1)]
        );
        assert_eq!(result.groups[0][1].start, 1);
        assert_eq!(result.groups[0][2].start, 3);
        assert_eq!(result.final_cycle, 9);
    }

    #[test]
    fn equal_alternative_resources_use_deterministic_lowest_index() {
        let result = schedule(
            &[
                Resource {
                    width: 1,
                    latency: 1,
                    initiation_interval: 1,
                    max_in_flight: None,
                },
                Resource {
                    width: 1,
                    latency: 1,
                    initiation_interval: 1,
                    max_in_flight: None,
                },
            ],
            &[vec![Task {
                resource: None,
                resource_options: vec![0, 1],
                work: 1,
                dependencies: vec![],
                cross_dependencies: vec![],
            }]],
            1,
        )
        .expect("equal-resource schedule should succeed");
        assert_eq!(result.selected_resources[0], vec![Some(0)]);
    }

    #[test]
    fn zero_work_alternatives_fail_closed() {
        assert!(matches!(
            schedule(
                &[Resource {
                    width: 1,
                    latency: 1,
                    initiation_interval: 1,
                    max_in_flight: None,
                }],
                &[vec![Task {
                    resource: None,
                    resource_options: vec![0],
                    work: 0,
                    dependencies: vec![],
                    cross_dependencies: vec![],
                }]],
                1,
            ),
            Err(PipelineError::Invalid(_))
        ));
    }

    #[test]
    fn overflowing_alternative_does_not_hide_a_viable_route() {
        let result = schedule(
            &[
                Resource {
                    width: 1,
                    latency: u64::MAX,
                    initiation_interval: 1,
                    max_in_flight: None,
                },
                Resource {
                    width: 1,
                    latency: 1,
                    initiation_interval: 1,
                    max_in_flight: None,
                },
            ],
            &[vec![Task {
                resource: None,
                resource_options: vec![0, 1],
                work: 1,
                dependencies: vec![],
                cross_dependencies: vec![],
            }]],
            1,
        )
        .expect("the representable alternative should be selected");
        assert_eq!(result.selected_resources[0], vec![Some(1)]);
    }

    #[test]
    fn overflowing_reservation_does_not_hide_a_viable_route() {
        let result = schedule(
            &[
                Resource {
                    width: 1,
                    latency: 1,
                    initiation_interval: u64::MAX,
                    max_in_flight: None,
                },
                Resource {
                    width: 1,
                    latency: 2,
                    initiation_interval: 1,
                    max_in_flight: None,
                },
            ],
            &[vec![
                Task {
                    resource: None,
                    resource_options: vec![],
                    work: 0,
                    dependencies: vec![],
                    cross_dependencies: vec![],
                },
                Task {
                    resource: None,
                    resource_options: vec![0, 1],
                    work: 1,
                    dependencies: vec![],
                    cross_dependencies: vec![],
                },
            ]],
            1,
        )
        .expect("a reservation overflow must not defeat the representable route");
        assert_eq!(result.selected_resources[0], vec![None, Some(1)]);
        assert_eq!(
            result.groups[0][1],
            Interval {
                start: 1,
                complete: 4
            }
        );
    }
}
