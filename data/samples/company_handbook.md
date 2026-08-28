# Acme Global Employee Handbook & Operating Guidelines

## 1. Welcome and Mission
Welcome to Acme Global! We build scalable, high-integrity AI and data systems. Our mission is to empower organizations with transparent, verifiable intelligence.

## 2. Remote Work & Core Hours
We are a remote-first company with team members across multiple time zones.
- Core Collaboration Hours: 10:00 AM to 3:00 PM EST. During these hours, employees should be reasonably reachable via Slack and available for scheduled standups.
- Home Office Stipend: Every full-time employee is eligible for a $1,500 initial home office setup stipend, followed by a $500 annual equipment refresh allowance.
- Internet Reimbursement: Up to $80/month for high-speed home internet connectivity.

## 3. Paid Time Off (PTO) & Leave Policy
Acme Global promotes a healthy work-life balance through flexible PTO:
- Standard PTO: Full-time employees receive 25 days of paid vacation per calendar year, accrued monthly.
- Sick & Mental Health Days: 10 dedicated days per year, no doctor's note required for absences under 3 consecutive days.
- Parental Leave: 16 weeks of 100% paid leave for all new parents (birth, adoption, or foster placement).
- Bereavement Leave: Up to 5 consecutive paid days for immediate family members.

## 4. Expense Reimbursement & Travel Policy
All business-related expenses must be submitted through Expensify within 30 days of the transaction.
- Daily Meal Allowance (Per Diem): Up to $85/day during approved business travel (Breakfast: $20, Lunch: $25, Dinner: $40).
- Flights: Economy class for flights under 6 hours; Premium Economy or Business class for flights exceeding 6 hours.
- Lodging: Standard hotel rooms up to $250/night in standard markets, or up to $350/night in high-cost metro areas (NYC, SF, London).
- Approval Thresholds: Any single expense over $1,000 requires pre-approval from your direct manager.

## 5. Engineering Standards & Code Quality
Our engineering culture values simplicity, testability, and deterministic behavior:
- Code Reviews: All pull requests require at least two passing peer reviews and 100% passing CI automated tests before merging to main.
- Verification: Every code change must be accompanied by behavior-driven unit and integration tests.
- Security Boundaries: Subprocess execution must strictly isolate environments, drop unneeded privileges, and enforce strict timeouts and memory limits.

## 6. Information Security & Compliance
Protecting customer data is everyone's responsibility:
- Multi-Factor Authentication (MFA): Mandatory on all corporate accounts and SSO logins using hardware keys or authenticator apps (SMS 2FA is prohibited).
- Device Encryption: All company-issued laptops must have FileVault/BitLocker disk encryption enabled with automatic screen lock set to 5 minutes or less.
- Incident Reporting: Any suspected security anomaly or credential leakage must be reported immediately to security@acmeglobal.internal within 1 hour.
