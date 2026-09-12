# Data

The three source CSVs are **not committed** to this repository — `flights.csv` alone is ~24 MB, and
raw data does not belong in a code repo. The trained artifact in `model/` is committed, so the web
app and the REST API run without them.

Put the files here to re-run the notebook end to end:

```
data/
├── flights.csv     271,888 rows
├── hotels.csv       40,552 rows
└── users.csv         1,340 rows
```

## Schema

**users.csv** — one row per employee
`code` (primary key) · `company` · `name` · `gender` · `age`

**flights.csv** — one row per flight leg booked
`travelCode` · `userCode` (→ `users.code`) · `from` · `to` · `flightType`
(`economic` / `premium` / `firstClass`) · `agency` (`CloudFy` / `Rainbow` / `FlyingDrops`) ·
`time` (hours) · `distance` (km) · `date` (MM/DD/YYYY) · **`price`** (target)

**hotels.csv** — one row per hotel stay
`travelCode` · `userCode` · `name` · `place` · `days` · `price` · `total` · `date`

`travelCode` links a flight to the hotel stay of the same trip; each trip has ~2 flight legs
(outbound + return).

## Using them in Colab

The notebook's loader searches the working directory, `/content`, any sub-folder, and any `.zip`
in `/content` — and falls back to an upload dialog. Just upload the three CSVs with the folder icon
in the Colab sidebar and run all cells.
