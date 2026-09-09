//! Observation-only Nsight Compute CSV ingestion for direct replay.
//!
//! These values are comparison targets. They never enter issue, memory, or
//! architecture configuration decisions.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct NcuResourceDependency {
    pub issue_instructions: Option<f64>,
    pub load_store_sectors: Option<f64>,
    pub eligible_warps: Option<f64>,
    pub scoreboard_stall_units: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct NcuObservation {
    pub launch_id: u64,
    pub kernel_name: String,
    pub device: String,
    pub block_size: Option<u64>,
    pub grid_size: Option<u64>,
    pub gpu_time_ns: Option<f64>,
    pub sm_cycles: Option<f64>,
    pub warp_instructions: Option<f64>,
    /// The .avg roll-up is kept separate because it is a mean across SMSP
    /// instances, not a total instruction count.
    pub warp_instructions_avg: Option<f64>,
    pub fma_instructions: Option<f64>,
    pub lsu_instructions: Option<f64>,
    pub eligible_warps: Option<f64>,
    pub long_scoreboard_stalls: Option<f64>,
    pub short_scoreboard_stalls: Option<f64>,
    pub global_load_sectors: Option<f64>,
    pub global_store_sectors: Option<f64>,
    pub resource_dependency: NcuResourceDependency,
}

impl NcuObservation {
    pub fn from_csv(text: &str, selected_launch: Option<u64>) -> Result<Self, NcuError> {
        let mut rows = text.lines().map(split_csv_line);
        let header = rows
            .find(|row| row.first().is_some_and(|field| field == "ID"))
            .ok_or(NcuError::MissingHeader)?;
        let mut candidates = Vec::new();
        for row in rows {
            let Some(id) = row.first().and_then(|field| field.parse::<u64>().ok()) else {
                continue;
            };
            if selected_launch.is_none_or(|wanted| wanted == id) {
                candidates.push((id, row));
            }
        }
        let (launch_id, data) = match candidates.len() {
            0 => {
                return Err(selected_launch.map_or(NcuError::MissingData, NcuError::LaunchNotFound));
            }
            1 => candidates.pop().ok_or(NcuError::MissingData)?,
            _ => return Err(NcuError::MultipleLaunches),
        };
        let value = |names: &[&str]| -> Result<Option<f64>, NcuError> {
            let Some(index) = names
                .iter()
                .find_map(|name| header.iter().position(|field| field == *name))
            else {
                return Ok(None);
            };
            let raw = data.get(index).ok_or(NcuError::ShortRow)?;
            if raw.is_empty() || raw.eq_ignore_ascii_case("N/A") {
                return Ok(None);
            }
            parse_number(raw)
                .map(Some)
                .ok_or_else(|| NcuError::InvalidNumber(names[0].to_owned(), raw.to_owned()))
        };
        let text_value = |names: &[&str]| -> Result<String, NcuError> {
            for name in names {
                let Some(index) = header.iter().position(|field| field == *name) else {
                    continue;
                };
                let raw = data.get(index).ok_or(NcuError::ShortRow)?;
                if !raw.is_empty() && !raw.eq_ignore_ascii_case("N/A") {
                    return Ok(raw.clone());
                }
            }
            Ok(String::new())
        };
        let integer = |name: &str| -> Result<Option<u64>, NcuError> {
            let Some(index) = header.iter().position(|field| field == name) else {
                return Ok(None);
            };
            let raw = data.get(index).ok_or(NcuError::ShortRow)?;
            if raw.is_empty() || raw.eq_ignore_ascii_case("N/A") {
                return Ok(None);
            }
            let normalized = raw.trim().replace(',', "");
            if let Ok(integer) = normalized.parse::<u64>() {
                return Ok(Some(integer));
            }
            // Do not accept decimal/exponent spellings through f64: values
            // above 2^53 can round to a different integer before conversion.
            // NCU's launch dimensions are emitted as decimal integers, so a
            // non-integral spelling is malformed rather than approximate.
            Err(NcuError::InvalidInteger(name.to_owned(), raw.to_owned()))
        };
        let global_load_sectors = value(&["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"])?;
        let global_store_sectors = value(&["l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"])?;
        let eligible_warps = value(&["smsp__warps_eligible.sum"])?;
        let long_scoreboard_stalls = value(&["smsp__warps_issue_stalled_long_scoreboard.sum"])?;
        let short_scoreboard_stalls = value(&["smsp__warps_issue_stalled_short_scoreboard.sum"])?;
        Ok(Self {
            launch_id,
            kernel_name: text_value(&["launch__kernel_name", "Kernel Name"])?,
            device: text_value(&["device__attribute_display_name"])?,
            block_size: integer("launch__block_size")?,
            grid_size: integer("launch__grid_size")?,
            gpu_time_ns: value(&["gpu__time_duration.avg"])?,
            sm_cycles: value(&["sm__cycles_elapsed.avg"])?,
            warp_instructions: value(&["smsp__inst_executed.sum"])?,
            warp_instructions_avg: value(&["smsp__inst_executed.avg"])?,
            fma_instructions: value(&["smsp__inst_executed_pipe_fma.sum"])?,
            lsu_instructions: value(&["smsp__inst_executed_pipe_lsu.sum"])?,
            eligible_warps,
            long_scoreboard_stalls,
            short_scoreboard_stalls,
            global_load_sectors,
            global_store_sectors,
            resource_dependency: NcuResourceDependency {
                issue_instructions: value(&["smsp__inst_executed.sum"])?,
                load_store_sectors: match (global_load_sectors, global_store_sectors) {
                    (Some(load), Some(store)) => Some(load + store),
                    _ => None,
                },
                eligible_warps,
                scoreboard_stall_units: match (long_scoreboard_stalls, short_scoreboard_stalls) {
                    (Some(long), Some(short)) => Some(long + short),
                    (Some(value), None) | (None, Some(value)) => Some(value),
                    (None, None) => None,
                },
            },
        })
    }
}

#[derive(Debug)]
pub enum NcuError {
    MissingHeader,
    MissingData,
    MultipleLaunches,
    LaunchNotFound(u64),
    ShortRow,
    InvalidNumber(String, String),
    InvalidInteger(String, String),
}

impl fmt::Display for NcuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingHeader => write!(f, "NCU CSV has no raw metric header"),
            Self::MissingData => write!(f, "NCU CSV has no numeric data row"),
            Self::MultipleLaunches => write!(f, "NCU CSV has multiple launches; select one by ID"),
            Self::LaunchNotFound(id) => write!(f, "NCU launch ID {id} was not found"),
            Self::ShortRow => write!(f, "NCU CSV data row is shorter than its header"),
            Self::InvalidNumber(name, value) => {
                write!(f, "NCU metric {name} is not numeric: {value}")
            }
            Self::InvalidInteger(name, value) => {
                write!(
                    f,
                    "NCU metric {name} is not a non-negative integer: {value}"
                )
            }
        }
    }
}

impl std::error::Error for NcuError {}

fn parse_number(raw: &str) -> Option<f64> {
    let value = raw.trim().replace(',', "").parse::<f64>().ok()?;
    value.is_finite().then_some(value)
}

fn split_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' if quoted && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                fields.push(field.trim().to_owned());
                field.clear();
            }
            _ => field.push(ch),
        }
    }
    fields.push(field.trim().to_owned());
    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csv(rows: &[&str]) -> String {
        let header = [
            "ID",
            "launch__kernel_name",
            "device__attribute_display_name",
            "launch__block_size",
            "launch__grid_size",
            "gpu__time_duration.avg",
        ]
        .join(",");
        format!("{}\n{}\n", header, rows.join("\n"))
    }

    #[test]
    fn parses_selected_launch_and_optional_missing_metrics() {
        let observation =
            NcuObservation::from_csv(&csv(&["7,kernel,device,256,4,123.5"]), Some(7)).unwrap();
        assert_eq!(observation.launch_id, 7);
        assert_eq!(observation.block_size, Some(256));
        assert_eq!(observation.grid_size, Some(4));
        assert_eq!(observation.gpu_time_ns, Some(123.5));
        assert_eq!(observation.global_load_sectors, None);
    }

    #[test]
    fn rejects_nonfinite_and_fractional_integer_metrics() {
        let nonfinite = csv(&["0,kernel,device,256,4,nan"]);
        assert!(matches!(
            NcuObservation::from_csv(&nonfinite, None),
            Err(NcuError::InvalidNumber(name, value))
                if name == "gpu__time_duration.avg" && value == "nan"
        ));

        let fractional = csv(&["0,kernel,device,32.5,4,1"]);
        assert!(matches!(
            NcuObservation::from_csv(&fractional, None),
            Err(NcuError::InvalidInteger(name, value))
                if name == "launch__block_size" && value == "32.5"
        ));

        let rounded = csv(&["0,kernel,device,9007199254740993.0,4,1"]);
        assert!(matches!(
            NcuObservation::from_csv(&rounded, None),
            Err(NcuError::InvalidInteger(name, value))
                if name == "launch__block_size" && value == "9007199254740993.0"
        ));
    }

    #[test]
    fn rejects_ambiguous_launch_selection() {
        let multiple = csv(&["0,kernel,device,256,4,1", "1,kernel,device,256,4,1"]);
        assert!(matches!(
            NcuObservation::from_csv(&multiple, None),
            Err(NcuError::MultipleLaunches)
        ));
        assert!(matches!(
            NcuObservation::from_csv(&multiple, Some(9)),
            Err(NcuError::LaunchNotFound(9))
        ));
    }

    #[test]
    fn raw_export_preserves_kernel_name_and_pipeline_observations() {
        let raw = "ID,Kernel Name,launch__kernel_name,smsp__inst_executed_pipe_fma.sum,smsp__inst_executed_pipe_lsu.sum\n0,\"sector_probe(unsigned*, unsigned*)\",,12,98304\n";
        let observation = NcuObservation::from_csv(raw, None).unwrap();
        assert_eq!(
            observation.kernel_name,
            "sector_probe(unsigned*, unsigned*)"
        );
        assert_eq!(observation.fma_instructions, Some(12.0));
        assert_eq!(observation.lsu_instructions, Some(98_304.0));
        assert_eq!(observation.warp_instructions, None);
        assert_eq!(observation.warp_instructions_avg, None);
        assert_eq!(observation.resource_dependency.issue_instructions, None);
    }

    #[test]
    fn keeps_instruction_sum_and_average_in_separate_units() {
        let raw = "ID,smsp__inst_executed.sum,smsp__inst_executed.avg\n0,100,25\n";
        let observation = NcuObservation::from_csv(raw, None).unwrap();
        assert_eq!(observation.warp_instructions, Some(100.0));
        assert_eq!(observation.warp_instructions_avg, Some(25.0));
        assert_eq!(
            observation.resource_dependency.issue_instructions,
            Some(100.0)
        );

        let avg_only = "ID,smsp__inst_executed.avg\n0,25\n";
        let observation = NcuObservation::from_csv(avg_only, None).unwrap();
        assert_eq!(observation.warp_instructions, None);
        assert_eq!(observation.warp_instructions_avg, Some(25.0));
        assert_eq!(observation.resource_dependency.issue_instructions, None);
    }
}
