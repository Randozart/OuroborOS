use serde::{Deserialize, Serialize};

/// Energy budget tracker for the cluster.
///
/// The budget is a first-class scheduling constraint:
/// no assignment is allowed if it would push total draw past the budget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyBudget {
    pub budget_watts: u32,
    pub current_watts: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BudgetCheck {
    Allowed,
    Exceeded { current: u32, budget: u32, requested: u32 },
}

impl EnergyBudget {
    pub fn new(budget_watts: u32) -> Self {
        Self {
            budget_watts,
            current_watts: 0,
        }
    }

    /// Check if adding `requested` watts stays within budget.
    pub fn check(&self, requested: u32) -> BudgetCheck {
        let projected = self.current_watts + requested;
        if projected <= self.budget_watts {
            BudgetCheck::Allowed
        } else {
            BudgetCheck::Exceeded {
                current: self.current_watts,
                budget: self.budget_watts,
                requested,
            }
        }
    }

    /// Commit a new draw (e.g., after dispatching a task).
    pub fn commit(&mut self, watts: u32) {
        self.current_watts += watts;
    }

    /// Release watts (e.g., after a task completes).
    pub fn release(&mut self, watts: u32) {
        self.current_watts = self.current_watts.saturating_sub(watts);
    }

    /// Set a new budget.
    pub fn set_budget(&mut self, watts: u32) {
        self.budget_watts = watts;
    }

    pub fn headroom(&self) -> u32 {
        self.budget_watts.saturating_sub(self.current_watts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_allows() {
        let mut budget = EnergyBudget::new(500);
        assert_eq!(budget.check(100), BudgetCheck::Allowed);
        budget.commit(100);
        assert_eq!(budget.check(400), BudgetCheck::Allowed);
    }

    #[test]
    fn test_budget_exceeds() {
        let mut budget = EnergyBudget::new(500);
        budget.commit(400);
        assert_eq!(
            budget.check(200),
            BudgetCheck::Exceeded {
                current: 400,
                budget: 500,
                requested: 200,
            }
        );
    }

    #[test]
    fn test_release() {
        let mut budget = EnergyBudget::new(500);
        budget.commit(300);
        budget.release(100);
        assert_eq!(budget.current_watts, 200);
        assert_eq!(budget.headroom(), 300);
    }

    #[test]
    fn test_saturating_release() {
        let mut budget = EnergyBudget::new(500);
        budget.commit(50);
        budget.release(500);
        assert_eq!(budget.current_watts, 0);
    }
}
