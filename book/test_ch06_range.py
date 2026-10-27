import unittest
from ch06_range import out_of_range

RANGES = dict(distance=dict(min=0, max=29), age=dict(min=0, max=58), area=dict(min=50, max=210))
STATIONS = ("池尻大橋", "三軒茶屋", "駒沢大学", "桜新町", "用賀", "二子玉川")
YEARS = ("2021", "2022", "2023", "2024", "2025")
ROOM = dict(distance=6, age=20, area=70, station="用賀", year=2025)


class RangeTests(unittest.TestCase):
    def test_built_year_input(self):
        room = {key: value for key, value in ROOM.items() if key != "age"}
        room["built_year"] = 2005
        self.assertEqual(out_of_range(room, RANGES, STATIONS, YEARS), [])
        self.assertNotIn("age", room)

    def test_built_year_takes_priority(self):
        self.assertTrue(out_of_range(dict(ROOM, age=20, built_year=1900), RANGES, STATIONS, YEARS))

    def test_inside(self):
        self.assertEqual(out_of_range(ROOM, RANGES, STATIONS, YEARS), [])

    def test_boundary_is_inside(self):
        room = dict(ROOM, distance=29, age=0, area=50)
        self.assertEqual(out_of_range(room, RANGES, STATIONS, YEARS), [])

    def test_outside(self):
        room = dict(ROOM, distance=30, area=49.9)
        self.assertEqual(len(out_of_range(room, RANGES, STATIONS, YEARS)), 2)

    def test_station_and_year(self):
        room = dict(ROOM, station="渋谷", year=2026)
        self.assertEqual(len(out_of_range(room, RANGES, STATIONS, YEARS)), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
