use std::collections::VecDeque;
use std::time::{Duration, Instant};

use super::Task;

/// A task waiting in the queue.
#[derive(Debug, Clone)]
pub struct QueuedTask {
    pub task: Task,
    pub enqueued_at: Instant,
    pub retries: u32,
    pub max_retries: u32,
    pub priority: u8,
    pub deadline: Option<Instant>,
}

impl QueuedTask {
    pub fn new(task: Task) -> Self {
        Self {
            task,
            enqueued_at: Instant::now(),
            retries: 0,
            max_retries: 3,
            priority: 0,
            deadline: None,
        }
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    pub fn with_max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    pub fn is_expired(&self) -> bool {
        self.deadline.map(|d| Instant::now() > d).unwrap_or(false)
    }

    pub fn age(&self) -> Duration {
        self.enqueued_at.elapsed()
    }
}

/// The task queue: holds tasks that couldn't be dispatched, retrying them
/// as nodes become available.
pub struct TaskQueue {
    pub tasks: VecDeque<QueuedTask>,
    max_size: usize,
}

impl TaskQueue {
    pub fn new() -> Self {
        Self {
            tasks: VecDeque::new(),
            max_size: 1000,
        }
    }

    pub fn with_max_size(mut self, max: usize) -> Self {
        self.max_size = max;
        self
    }

    /// Enqueue a task. Returns true if accepted, false if queue is full.
    pub fn enqueue(&mut self, task: Task) -> bool {
        if self.tasks.len() >= self.max_size {
            return false;
        }
        self.tasks.push_back(QueuedTask::new(task));
        true
    }

    /// Enqueue with priority (higher = more urgent).
    pub fn enqueue_priority(&mut self, task: Task, priority: u8) -> bool {
        if self.tasks.len() >= self.max_size {
            return false;
        }
        let qt = QueuedTask::new(task).with_priority(priority);
        // Insert before the first lower-priority item
        let pos = self.tasks.iter().position(|t| t.priority < priority).unwrap_or(self.tasks.len());
        self.tasks.insert(pos, qt);
        true
    }

    /// Pop the highest-priority non-expired task.
    pub fn pop(&mut self) -> Option<QueuedTask> {
        // Remove expired tasks at the front
        while let Some(front) = self.tasks.front() {
            if front.is_expired() {
                self.tasks.pop_front();
            } else {
                break;
            }
        }
        self.tasks.pop_front()
    }

    /// Peek at the next task without removing it.
    pub fn peek(&self) -> Option<&QueuedTask> {
        self.tasks.iter().find(|t| !t.is_expired())
    }

    /// Mark a task as retried (increment retry count, re-enqueue).
    pub fn retry(&mut self, task_id: &str) -> bool {
        if let Some(pos) = self.tasks.iter().position(|t| t.task.name == task_id) {
            let mut qt = self.tasks.remove(pos).unwrap();
            qt.retries += 1;
            if qt.retries <= qt.max_retries {
                // Re-insert at back (lower priority after retry)
                self.tasks.push_back(qt);
                return true;
            }
        }
        false
    }

    /// Remove a task by name.
    pub fn remove(&mut self, task_name: &str) -> Option<QueuedTask> {
        if let Some(pos) = self.tasks.iter().position(|t| t.task.name == task_name) {
            self.tasks.remove(pos)
        } else {
            None
        }
    }

    /// Drain all tasks (for rescheduling).
    pub fn drain(&mut self) -> Vec<QueuedTask> {
        self.tasks.drain(..).collect()
    }

    /// Number of tasks in queue.
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Is queue empty.
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Summary: task names, ages, retry counts.
    pub fn summary(&self) -> Vec<QueueEntry> {
        self.tasks.iter().map(|qt| QueueEntry {
            name: qt.task.name.clone(),
            class: format!("{:?}", qt.task.class),
            age_secs: qt.age().as_secs(),
            retries: qt.retries,
            priority: qt.priority,
        }).collect()
    }

    /// Remove expired tasks and return them.
    pub fn expire(&mut self) -> Vec<QueuedTask> {
        let mut expired = Vec::new();
        let mut i = 0;
        while i < self.tasks.len() {
            if self.tasks[i].is_expired() {
                expired.push(self.tasks.remove(i).unwrap());
            } else {
                i += 1;
            }
        }
        expired
    }
}

impl Default for TaskQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary entry for display.
#[derive(Debug, Clone)]
pub struct QueueEntry {
    pub name: String,
    pub class: String,
    pub age_secs: u64,
    pub retries: u32,
    pub priority: u8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::workload_class::WorkloadClass;

    fn make_task(name: &str, class: WorkloadClass) -> Task {
        Task {
            name: name.to_string(),
            class,
            payload: String::new(),
            estimated_watts: 10,
            estimated_seconds: 5,
        }
    }

    #[test]
    fn test_enqueue_and_pop() {
        let mut q = TaskQueue::new();
        assert!(q.enqueue(make_task("t1", WorkloadClass::SimdFriendly)));
        assert_eq!(q.len(), 1);
        let qt = q.pop().unwrap();
        assert_eq!(qt.task.name, "t1");
        assert!(q.is_empty());
    }

    #[test]
    fn test_priority_ordering() {
        let mut q = TaskQueue::new();
        q.enqueue_priority(make_task("low", WorkloadClass::Unknown), 0);
        q.enqueue_priority(make_task("high", WorkloadClass::Unknown), 10);
        q.enqueue_priority(make_task("mid", WorkloadClass::Unknown), 5);
        assert_eq!(q.pop().unwrap().task.name, "high");
        assert_eq!(q.pop().unwrap().task.name, "mid");
        assert_eq!(q.pop().unwrap().task.name, "low");
    }

    #[test]
    fn test_retry_limit() {
        let mut q = TaskQueue::new();
        let mut task = make_task("t1", WorkloadClass::SimdFriendly);
        q.enqueue(task);
        // Simulate 4 retries (max_retries=3)
        assert!(q.retry("t1"));
        assert!(q.retry("t1"));
        assert!(q.retry("t1"));
        assert!(!q.retry("t1")); // 4th retry fails
    }

    #[test]
    fn test_remove() {
        let mut q = TaskQueue::new();
        q.enqueue(make_task("t1", WorkloadClass::SimdFriendly));
        q.enqueue(make_task("t2", WorkloadClass::SimdFriendly));
        let removed = q.remove("t1").unwrap();
        assert_eq!(removed.task.name, "t1");
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_max_size() {
        let mut q = TaskQueue::new().with_max_size(2);
        assert!(q.enqueue(make_task("t1", WorkloadClass::Unknown)));
        assert!(q.enqueue(make_task("t2", WorkloadClass::Unknown)));
        assert!(!q.enqueue(make_task("t3", WorkloadClass::Unknown)));
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_drain() {
        let mut q = TaskQueue::new();
        q.enqueue(make_task("t1", WorkloadClass::Unknown));
        q.enqueue(make_task("t2", WorkloadClass::Unknown));
        let tasks = q.drain();
        assert_eq!(tasks.len(), 2);
        assert!(q.is_empty());
    }

    #[test]
    fn test_summary() {
        let mut q = TaskQueue::new();
        q.enqueue_priority(make_task("t1", WorkloadClass::SimdFriendly), 5);
        let s = q.summary();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].name, "t1");
        assert_eq!(s[0].priority, 5);
    }
}
