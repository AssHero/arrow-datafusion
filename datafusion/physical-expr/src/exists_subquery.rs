// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
/// Exists/Not Exists subquery expression
use crate::physical_expr::down_cast_any_ref;
use crate::PhysicalExpr;

use arrow::{
    array::{ArrayData, BooleanArray},
    buffer::MutableBuffer,
    datatypes::{DataType, Schema},
    record_batch::RecordBatch,
};
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::{
    ColumnarValue, Limit, LogicalPlan, PlanType, Subquery, ToStringifiedPlan,
};
use std::collections::HashSet;
use std::fmt::Debug;
use std::{any::Any, sync::Arc};

/// Exists/Not Exists subquery expression
///
/// We only support uncorrelated exists subquery here, correlated exists subquery
/// should be transformed to semi/inner join in logical optimizer.
///
/// input is the subquery to be executed
///
/// negated indicates the subquery is exists or not exists
///
/// executed indicates that the subquery has been executed or not, uncorrelated exists subquery
/// should be executed only once, and store the result(true or false).
///
/// result is the result of this uncorrelated exists subquery.
///
/// For uncorrelated exists subquery, if rows returned from input, result is true; otherwise, result is false.
/// For uncorrelated not exists subquery, if no rows returned from input, result is true; otherwise, result is false.
///
/// We execute the input once, store the result. Everytime we do the expr evaluation, if result is true, return all records;
/// otherwise, return none.
#[derive(Debug)]
pub struct ExistsSubqueryExpr {
    input: Arc<LogicalPlan>,
    negated: bool,
    executed: bool,
    result: bool,
}

impl ExistsSubqueryExpr {
    /// Create a new ExistsSubqueryExpr
    pub fn new(
        input: Arc<LogicalPlan>,
        negated: bool,
        executed: bool,
        result: bool,
    ) -> Self {
        Self {
            input,
            negated,
            executed,
            result,
        }
    }

    /// Get the input Logical plan
    pub fn input(&self) -> &Arc<LogicalPlan> {
        &self.input
    }

    /// Get negated of exists subquery
    pub fn negated(&self) -> bool {
        self.negated
    }
}

impl std::fmt::Display for ExistsSubqueryExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.negated {
            write!(f, "NOT EXISTS (<subquery>)
                        subquery:
                            {:?}", self.input())
        } else {
            write!(f, "EXISTS (<subquery>)
                        subquery:
                            {:?}", self.input())
        }
    }
}

impl PhysicalExpr for ExistsSubqueryExpr {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let num_rows = batch.num_rows();

        // the subquery should be executed before.
        if !self.executed {
            return Err(DataFusionError::Internal(
                "Unsupported exists subquery".to_string(),
            ));
        }

        let output = if self.result {
            MutableBuffer::new(num_rows)
                .with_bitset(num_rows, true)
                .into()
        } else {
            MutableBuffer::from_len_zeroed(num_rows).into()
        };

        let data = unsafe {
            ArrayData::new_unchecked(
                DataType::Boolean,
                num_rows,
                None,
                None,
                0,
                vec![output],
                vec![],
            )
        };

        Ok(ColumnarValue::Array(Arc::new(BooleanArray::from(data))))
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(self)
    }
}

impl PartialEq<dyn Any> for ExistsSubqueryExpr {
    fn eq(&self, other: &dyn Any) -> bool {
        down_cast_any_ref(other)
            .downcast_ref::<Self>()
            .map(|x| {
                let stringifiel_plan_self =
                    self.input.to_stringified(PlanType::FinalLogicalPlan);
                let stringifiel_plan_other =
                    x.input.to_stringified(PlanType::FinalLogicalPlan);

                self.negated == x.negated
                    && stringifiel_plan_self.plan == stringifiel_plan_other.plan
            })
            .unwrap_or(false)
    }
}

/// Create an EXISTS SUBQUERY expression
/// For uncorrelated exists subquery, we only need one rows, add the limit logical plan to the input.
pub fn exists_subquery(
    arg: &Arc<LogicalPlan>,
    negated: bool,
    executed: bool,
    result: bool,
) -> Result<Arc<dyn PhysicalExpr>> {
    let limit_plan = LogicalPlan::Limit(Limit {
        skip: 0,
        fetch: Some(1),
        input: Arc::new(arg.as_ref().clone()),
    });
    Ok(Arc::new(ExistsSubqueryExpr::new(
        Arc::new(limit_plan),
        negated,
        executed,
        result,
    )))
}

/// is this subquery an uncorrelated subquery?
pub fn is_uncorrelated_subquery(subquery: &Subquery) -> Result<bool> {
    let schema_columns = subquery
        .subquery
        .schema()
        .fields()
        .iter()
        .flat_map(|f| [f.qualified_column(), f.unqualified_column()])
        .collect::<HashSet<_>>();

    let filter_columns = subquery.subquery.filter_columns()?;

    for cols in filter_columns {
        if schema_columns
            .intersection(&cols)
            .collect::<HashSet<_>>()
            .len()
            != cols.len()
        {
            return Ok(false);
        }
    }

    Ok(true)
}
