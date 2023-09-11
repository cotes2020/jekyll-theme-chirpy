# Loan details
purchase_price = 940000
down_payment = 188000
loan_amount = 752000

# interest_rate = 0.06625  # 6.625% annual rate

interest_rate = 0.0675  # 6.625% annual rate
# interest_rate = 0.04  # 6.625% annual rate

loan_term_months = 360
monthly_payment = 4811

additional_payment_25th_month = 242000
additional_payment_36th_month = 60000

yearly_additional_payment = 60000
years_for_additional_payment = [48, 60, 72, 84]

# Initialize balances
current_balance = loan_amount
remaining_months = loan_term_months

yearly_interest_payment = 0
total_interest_payment = 0

# Print header
print(
    "| Month | Starting Balance | Monthly Payment | Interest Payment | Principal Payment | Additional Payment | Ending Balance |"
)
print(
    "| ----- | ---------------- | --------------- | ---------------- | ----------------- | ------------------ | -------------- |"
)

# Print each month's details
for month in range(1, loan_term_months + 1):
    interest_payment = current_balance * interest_rate / 12
    principal_payment = monthly_payment - interest_payment

    # Apply additional payments at the specified months
    if month == 24:
        current_balance -= additional_payment_25th_month
    elif month == 36:
        current_balance -= additional_payment_36th_month
    elif month in years_for_additional_payment:
        # print("hhhhh")
        # print(current_balance - yearly_additional_payment)
        # print(current_balance)
        # print(yearly_additional_payment)
        current_balance -= yearly_additional_payment

    ending_balance = current_balance - principal_payment

    # Print the row for this month
    print(
        f"| {month:5d} | ${current_balance:,.2f} | ${monthly_payment:,.2f} | ${interest_payment:,.2f} | ${principal_payment:,.2f} | ${additional_payment_25th_month if month == 24 else additional_payment_36th_month if month == 36 else yearly_additional_payment if month in years_for_additional_payment else 0:,.2f} | ${ending_balance:,.2f} |"
    )

    current_balance = ending_balance

    if month in [13, 25, 37, 49, 61, 73, 85]:
        print(f"yearly_interest_payment: ${month} --> ${yearly_interest_payment:,.2f}")
        yearly_interest_payment = 0

    yearly_interest_payment += interest_payment

    total_interest_payment += interest_payment

    if ending_balance < 0:
        break

# Print footer
print(f"total_interest_payment: ${total_interest_payment:,.2f}")
