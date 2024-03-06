# Loan details
purchase_price = 940000
down_payment = 188000
# loan_amount = 752000
loan_amount = 744000

interest_rate = 0.0675  # 6.75% annual rate
# interest_rate = 0.0625
# interest_rate = 0.04

loan_term_months = 360
monthly_payment = 4825.57


target_month = 24
# additional_payment_at_target_month = 242000
additional_payment_at_target_month = 0


yearly_additional_payment = 2288
# yearly_additional_payment = 90000
# yearly_additional_payment = 60000
# yearly_additional_payment = 0
# yearly_additional_payment = 30000
years_for_additional_payment = [2]
# years_for_additional_payment = [
#     12,
#     24,
#     36,
#     48,
#     60,
#     72,
#     84,
#     96,
#     108,
#     120,
#     132,
#     144,
#     156,
#     268,
#     180,
#     192,
#     204,
#     216,
# ]


# Initialize balances
current_balance = loan_amount
remaining_months = loan_term_months

yearly_interest_payment = 0
total_interest_payment = 0
year = 0

# # Print header
# print(
#     "| Month | Starting Balance | Monthly Payment | Interest Payment | Principal Payment | Additional Payment | Ending Balance |"
# )
# print(
#     "| ----- | ---------------- | --------------- | ---------------- | ----------------- | ------------------ | -------------- |"
# )

# Print each month's details
for month in range(1, loan_term_months + 1):
    interest_payment = current_balance * interest_rate / 12
    principal_payment = monthly_payment - interest_payment

    # Apply additional payments at the specified months
    if month == target_month:
        current_balance -= additional_payment_at_target_month
    elif month in years_for_additional_payment:
        # print("hhhhh")
        # print(current_balance - yearly_additional_payment)
        # print(current_balance)
        # print(yearly_additional_payment)
        current_balance -= yearly_additional_payment

    ending_balance = current_balance - principal_payment

    # # Print the row for this month
    # print(
    #     f"| {month:5d} | ${current_balance:,.2f} | ${monthly_payment:,.2f} | ${interest_payment:,.2f} | ${principal_payment:,.2f} | ${additional_payment_at_target_month if month == 24 else yearly_additional_payment if month in years_for_additional_payment else 0:,.2f} | ${ending_balance:,.2f} |"
    # )

    current_balance = ending_balance

    if month % 12 == 0:
        year += 1
        print(
            f"{year}-{month} yearly_interest_payment: --> {yearly_interest_payment:,.2f}, current_balance: --> {current_balance}"
        )
        yearly_interest_payment = 0
    # if month in [13, 25, 37, 49, 61, 73, 85]:
    #     print(f"yearly_interest_payment: ${month} --> ${yearly_interest_payment:,.2f}")
    #     yearly_interest_payment = 0

    yearly_interest_payment += interest_payment

    total_interest_payment += interest_payment

    if ending_balance < 0:
        break

# Print footer
print(f"total_interest_payment: ${total_interest_payment:,.2f}")
