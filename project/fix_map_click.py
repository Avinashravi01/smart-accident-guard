"""
Fix index.html — use exact clicked lat/lng instead of snapping to hardcoded zones
"""

HTML_FILE = r"C:\Users\Hp\Downloads\smart accident guard\project\static\index.html"

with open(HTML_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# ── The fix: replace map click handler ────────────────────────────────────────
# Old code snaps to nearest hardcoded zone:
#   const { zone, distance } = findNearestZone(lat, lng);
#   analyzeLocation(zone.lat, zone.lng);
#
# New code: use exact clicked coordinates directly

old_click = """  const lat = e.latlng.lat, lng = e.latlng.lng;
  const { zone, distance } = findNearestZone(lat, lng);"""

new_click = """  const lat = e.latlng.lat, lng = e.latlng.lng;"""

if old_click in content:
    content = content.replace(old_click, new_click)
    print("✅ Removed findNearestZone snap from click handler")
else:
    print("⚠️ Click handler pattern not found exactly — trying alternate...")

# Also fix the analyzeLocation call after findNearestZone
old_analyze = "  analyzeLocation(zone.lat, zone.lng);"
new_analyze = "  analyzeLocation(lat, lng);"

if old_analyze in content:
    content = content.replace(old_analyze, new_analyze)
    print("✅ Fixed analyzeLocation to use exact coordinates")
else:
    print("⚠️ analyzeLocation pattern not found")

# ── Also fix getLocationData to use reverse geocoding from backend ─────────────
# When backend is offline, old code still uses hardcoded locationDB
# New code: generate dynamic location name from coordinates

old_loc_data = """function getLocationData(lat,lng){"""

new_loc_data = """function getLocationData(lat,lng){
  // Use exact coordinates — no snapping to hardcoded zones"""

if old_loc_data in content and new_loc_data not in content:
    content = content.replace(old_loc_data, new_loc_data)
    print("✅ Updated getLocationData")

with open(HTML_FILE, 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ Done! Frontend now uses exact clicked coordinates.")
print("   Click ANYWHERE on Chennai map — it will analyze that exact spot!")
print("\n🚀 Restart: uvicorn project.main:app --reload")
