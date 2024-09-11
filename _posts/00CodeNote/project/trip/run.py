import csv
from datetime import datetime, timedelta


# Function to create an event
def create_event(name, location, start_date, duration_hours=3):
    return {
        "Subject": name,
        "Location": location,
        "Start Date": start_date.strftime("%m/%d/%Y"),
        "Start Time": start_date.strftime("%I:%M %p"),
        "End Date": (start_date + timedelta(hours=duration_hours)).strftime("%m/%d/%Y"),
        "End Time": (start_date + timedelta(hours=duration_hours)).strftime("%I:%M %p"),
    }


# Define the start date of the trip
trip_start = datetime(2024, 11, 8)

# List to hold all events
events = []

# Day 1: Shibuya
events.append(
    create_event(
        "Shibuya Crossing",
        "Shibuya City, Tokyo 150-8010, Japan",
        trip_start + timedelta(hours=15),
    )
)
events.append(
    create_event(
        "Shibuya Scramble Square",
        "2 Chome-24-12 Shibuya, Shibuya City, Tokyo 150-0002, Japan",
        trip_start + timedelta(hours=17),
    )
)
events.append(
    create_event(
        "Hachiko Statue",
        "2 Chome-1 Dogenzaka, Shibuya City, Tokyo 150-0043, Japan",
        trip_start + timedelta(hours=20),
    )
)

# Day 2: Asakusa and Sumida
events.append(
    create_event(
        "Senso-ji Temple",
        "2 Chome-3-1 Asakusa, Taito City, Tokyo 111-0032, Japan",
        trip_start + timedelta(days=1, hours=9),
    )
)
events.append(
    create_event(
        "Sumida Park",
        "1 Chome Mukojima, Sumida City, Tokyo 131-0033, Japan",
        trip_start + timedelta(days=1, hours=12),
    )
)
events.append(
    create_event(
        "Tokyo Skytree",
        "1 Chome-1-2 Oshiage, Sumida City, Tokyo 131-0045, Japan",
        trip_start + timedelta(days=1, hours=15),
    )
)

# Day 3: Harajuku and Shinjuku
events.append(
    create_event(
        "Meiji Shrine",
        "1-1 Yoyogikamizonocho, Shibuya City, Tokyo 151-8557, Japan",
        trip_start + timedelta(days=2, hours=9),
    )
)
events.append(
    create_event(
        "Takeshita Street",
        "1 Chome-17 Jingumae, Shibuya City, Tokyo 150-0001, Japan",
        trip_start + timedelta(days=2, hours=12),
    )
)
events.append(
    create_event(
        "Omotesando Avenue",
        "Jingumae, Shibuya City, Tokyo 150-0001, Japan",
        trip_start + timedelta(days=2, hours=14),
    )
)
events.append(
    create_event(
        "Shinjuku Gyoen National Garden",
        "11 Naitomachi, Shinjuku City, Tokyo 160-0014, Japan",
        trip_start + timedelta(days=2, hours=16),
    )
)
events.append(
    create_event(
        "Omoide Yokocho",
        "1 Chome-2 Nishishinjuku, Shinjuku City, Tokyo 160-0023, Japan",
        trip_start + timedelta(days=2, hours=19),
    )
)

# Day 4: Nikko Day Trip
events.append(
    create_event(
        "Toshogu Shrine",
        "2301 Sannai, Nikko, Tochigi 321-1431, Japan",
        trip_start + timedelta(days=3, hours=9),
        duration_hours=10,
    )
)

# Day 5: Tsukiji and Ginza
events.append(
    create_event(
        "Tsukiji Outer Market",
        "4 Chome-16-2 Tsukiji, Chuo City, Tokyo 104-0045, Japan",
        trip_start + timedelta(days=4, hours=9),
    )
)
events.append(
    create_event(
        "Hamarikyu Gardens",
        "1-1 Hamarikyuteien, Chuo City, Tokyo 104-0046, Japan",
        trip_start + timedelta(days=4, hours=12),
    )
)
events.append(
    create_event(
        "Ginza",
        "Ginza, Chuo City, Tokyo 104-0061, Japan",
        trip_start + timedelta(days=4, hours=15),
    )
)

# Day 6: Odaiba and Roppongi
events.append(
    create_event(
        "TeamLab Borderless",
        "1 Chome-3-8 Aomi, Koto City, Tokyo 135-0064, Japan",
        trip_start + timedelta(days=5, hours=9),
    )
)
events.append(
    create_event(
        "DiverCity Tokyo Plaza",
        "1 Chome-1-10 Aomi, Koto City, Tokyo 135-0064, Japan",
        trip_start + timedelta(days=5, hours=12),
    )
)
events.append(
    create_event(
        "Odaiba Marine Park",
        "1 Chome-4 Daiba, Minato City, Tokyo 135-0091, Japan",
        trip_start + timedelta(days=5, hours=15),
    )
)
events.append(
    create_event(
        "Roppongi Hills",
        "6 Chome-10-1 Roppongi, Minato City, Tokyo 106-6108, Japan",
        trip_start + timedelta(days=5, hours=18),
    )
)

# Day 7: Akihabara and Ueno
events.append(
    create_event(
        "Akihabara",
        "Sotokanda, Chiyoda City, Tokyo 101-0021, Japan",
        trip_start + timedelta(days=6, hours=9),
    )
)
events.append(
    create_event(
        "Ameya-Yokocho",
        "4 Chome-7-8 Ueno, Taito City, Tokyo 110-0005, Japan",
        trip_start + timedelta(days=6, hours=12),
    )
)
events.append(
    create_event(
        "Ueno Park",
        "Uenokoen, Taito City, Tokyo 110-0007, Japan",
        trip_start + timedelta(days=6, hours=15),
    )
)
events.append(
    create_event(
        "Tokyo National Museum",
        "13-9 Uenokoen, Taito City, Tokyo 110-8712, Japan",
        trip_start + timedelta(days=6, hours=18),
    )
)

# Day 8: Final Day in Tokyo
events.append(
    create_event(
        "Shopping in Shibuya or Harajuku",
        "Shibuya or Harajuku, Tokyo, Japan",
        trip_start + timedelta(days=7, hours=9),
    )
)
events.append(
    create_event(
        "Relaxing Onsen Experience",
        "Odaiba, Tokyo, Japan",
        trip_start + timedelta(days=7, hours=14),
    )
)

# Write the events to a CSV file
with open("tokyo_trip.csv", mode="w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Subject",
            "Location",
            "Start Date",
            "Start Time",
            "End Date",
            "End Time",
        ],
    )
    writer.writeheader()
    for event in events:
        writer.writerow(event)

print("CSV file 'tokyo_trip.csv' created successfully!")
