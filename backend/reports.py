"""
Farm verification report, for lenders, insurers and buyers: the farm is there, it is this big, it was cropped in
these seasons (from the satellites), it had this much rain against normal, and (where the land is on GeoEstate Land
Registry) its title status. Numbered and kept as issued, so anyone holding it can check it at /verify.html.

It is evidence from satellites, weather models and registers, each with its limits written on the report; it is not a
field inspection or a title search.
"""
import io
import os
import secrets
from datetime import date, datetime, timezone

import httpx

CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def new_code() -> str:
    return "FSV-" + "".join(secrets.choice(CODE_ALPHABET) for _ in range(8))


def title_status(lat: float, lng: float) -> dict | None:
    """What GeoEstate Land Registry's published registries hold at the point (LAND_API + LAND_PLATFORM_KEY)."""
    api, key = os.environ.get("LAND_API", "").rstrip("/"), os.environ.get("LAND_PLATFORM_KEY", "")
    if not (api and key):
        return None
    try:
        r = httpx.get(f"{api}/platform/title-status", params={"lat": lat, "lng": lng}, headers={"X-Platform-Key": key}, timeout=10)
        r.raise_for_status()
        b = r.json()
        return {"ok": True, "registries_searched": b.get("registries_searched", 0), "at": b.get("at", []), "near": b.get("near", [])}
    except Exception:
        return {"ok": False}


def rainfall(history: dict | None, climate: dict | None) -> dict | None:
    """Rain in each of the last 3 complete years against the 10-year average, from the daily history."""
    if not history or not history.get("dates"):
        return None
    totals: dict[int, list] = {}
    for d, r in zip(history["dates"], history.get("rain") or []):
        totals.setdefault(int(d[:4]), []).append(r)
    years = {y: sum(v for v in vals if v is not None) for y, vals in totals.items()
             if sum(v is not None for v in vals) >= 330}
    if not years:
        return None
    normal = sum(years.values()) / len(years)
    last = sorted(years)[-3:]
    return {"normal_mm": round(normal), "normal_years": f"{min(years)}–{max(years)}", "source": history.get("source"),
            "years": [{"year": y, "rain_mm": round(years[y]), "vs_normal_pct": round(100 * (years[y] - normal) / normal)} for y in last],
            "wet_months": (climate or {}).get("wet_months"),
            "note": "Modelled for the area (~9–25 km), not measured on the farm."}


def build_pdf(c: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Flowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    NAVY, GREEN, MUTED = colors.HexColor("#1b2b5e"), colors.HexColor("#1f5c33"), colors.HexColor("#5c6b62")
    ss = getSampleStyleSheet()
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=NAVY, fontSize=12.5, spaceBefore=8, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=10.5, textColor=MUTED)
    farm = c["farm"]

    class Outline(Flowable):
        """The farm's boundary, drawn to scale, with a north arrow and a scale bar."""
        def __init__(self, geometry, size=55 * mm):
            super().__init__()
            self.g, self.size = geometry, size
            self.width = self.height = size

        def draw(self):
            import math
            rings = self.g["coordinates"] if self.g["type"] == "Polygon" else [r for p in self.g["coordinates"] for r in p]
            pts = [p for r in rings for p in r]
            lat0 = sum(p[1] for p in pts) / len(pts)
            k = math.cos(math.radians(lat0))
            xs, ys = [p[0] * k for p in pts], [p[1] for p in pts]
            span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1e-9
            pad = 6 * mm
            sc = (self.size - 2 * pad) / span
            cv = self.canv
            cv.setStrokeColor(GREEN); cv.setFillColor(colors.HexColor("#eaf6ee")); cv.setLineWidth(1.4)
            for r in rings:
                path = cv.beginPath()
                for i, (x, y) in enumerate(r):
                    px, py = pad + (x * k - min(xs)) * sc, pad + (y - min(ys)) * sc
                    path.moveTo(px, py) if i == 0 else path.lineTo(px, py)
                path.close()
                cv.drawPath(path, stroke=1, fill=1)
            cv.setFillColor(NAVY); cv.setFont("Helvetica-Bold", 8)
            cv.drawString(self.size - 6 * mm, self.size - 6 * mm, "N")
            metres = span * 111320
            bar = next(b for b in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000) if b >= metres / 4)
            cv.setLineWidth(1); cv.setStrokeColor(NAVY)
            cv.line(pad, 2 * mm, pad + bar / 111320 * sc, 2 * mm)
            cv.setFont("Helvetica", 7)
            cv.drawString(pad, 3 * mm, f"{bar:,} m")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title=f"Farm verification {c['code']}", author="GeoEstate FarmScan")
    tbl = lambda rows, widths: Table([[Paragraph(str(a), body), Paragraph(str(b), body)] for a, b in rows], colWidths=widths,  # noqa: E731
                                     style=TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#dde5df")),
                                                       ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3)]))
    story = [Paragraph(f"<font color='#1f5c33'><b>GeoEstate FarmScan</b></font> · Farm verification report", ss["Title"]),
             Paragraph(f"<b>{c['code']}</b> · issued {c['issued_on']} to {c['org_name']} · check it at {c['verify_url']}", small),
             Spacer(1, 6)]
    rows = [("Farm", farm["name"]), ("Measured area", f"{farm['area_ha']:,.2f} ha (walked or drawn boundary, measured on the ellipsoid)")]
    if farm.get("declared_ha"):
        rows.append(("Size the farmer gives", f"{farm['declared_ha']} ha ({farm['declared_vs_measured_pct']:+d}% against the measured area)"))
    rows += [("Where", ", ".join(x for x in [farm.get("community"), farm.get("lga"), farm.get("state")] if x) or "—"),
             ("Centre", f"{farm['lat']:.5f}, {farm['lng']:.5f}")]
    if farm.get("farmer"):
        rows.append(("Farmer (as recorded)", farm["farmer"]))
    if farm.get("crops"):
        rows.append(("Crops (as recorded)", ", ".join(farm["crops"])))
    if farm.get("mapped_by"):
        rows.append(("Mapped in the field", f"by {farm['mapped_by']} (GeoEstate Geohub)"))
    story += [Table([[tbl(rows, [38 * mm, 82 * mm]), Outline(c["geometry"])]], colWidths=[122 * mm, 58 * mm],
                    style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))]

    story.append(Paragraph("Crop seasons seen from the satellites", h2))
    if c["seasons"]:
        srows = [["Year", "Sown (estimate)", "Green-up", "Peak", "Harvest / dry-down", "From"]]
        for s in c["seasons"]:
            if not s.get("based_on"):
                srows.append([s["year"], Paragraph(s.get("why") or "No clear crop cycle", small), "", "", "", ""])
                continue
            pm = lambda x: f"{x['date']} ±{x['plus_minus_days']} d" if x and x.get("plus_minus_days") is not None else (x or {}).get("date", "—")  # noqa: E731
            srows.append([s["year"], f"{s['planting']['from']} – {s['planting']['to']}" if s.get("planting") else "before the first image",
                          pm(s.get("greenup")), pm(s.get("peak")), pm(s.get("end")) if s.get("end") else "not yet",
                          "radar" if s["based_on"] == "rvi" else "optical"])
        story.append(Table(srows, colWidths=[14 * mm, 40 * mm, 32 * mm, 32 * mm, 34 * mm, 18 * mm], hAlign="LEFT", style=TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#dde5df")), ("VALIGN", (0, 0), (-1, -1), "TOP")])))
        story.append(Paragraph(f"From {c['images']['optical']} cloud-free Sentinel-2 images and {c['images']['radar']} Sentinel-1 "
                               "radar passes in the farm's timeline. Green-up is when the crop's greenness rose through 20% of the "
                               "season's range, harvest when it fell back through 50%; ± is half the gap between the images around "
                               "it. Sowing is placed 1–4 weeks before green-up.", small))
    else:
        story.append(Paragraph("No season could be read: the farm's satellite timeline is empty for these years. "
                               "Build the season in FarmScan before issuing a report.", body))

    story.append(Paragraph("Rainfall", h2))
    rf = c.get("rainfall")
    if rf:
        story.append(Table([["Year", "Rain", "Against the average"]] +
                           [[y["year"], f"{y['rain_mm']:,} mm", f"{y['vs_normal_pct']:+d}%"] for y in rf["years"]],
                           colWidths=[25 * mm, 35 * mm, 40 * mm], hAlign="LEFT",
                           style=TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                             ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#dde5df"))])))
        story.append(Paragraph(f"Average {rf['normal_mm']:,} mm a year ({rf['normal_years']}). {rf['note']} Source: {rf.get('source') or 'Open-Meteo'}.", small))
    else:
        story.append(Paragraph("Rainfall could not be fetched when this report was issued.", body))

    story.append(Paragraph("Land register", h2))
    t = c.get("title")
    if t is None:
        story.append(Paragraph("Not checked: GeoEstate Land Registry is not connected to this FarmScan.", body))
    elif not t.get("ok"):
        story.append(Paragraph("GeoEstate Land Registry could not be reached when this report was issued.", body))
    elif t["at"]:
        for p in t["at"]:
            title = p.get("title")
            enc = ", ".join(e["kind"] for e in p.get("encumbrances", [])) or "none registered"
            story.append(Paragraph(f"<b>{p['pin']}</b> in {p['registry']}: "
                                   f"{(title['kind'] + ' ' + title['title_no']) if title else 'registered, no title'}; "
                                   f"encumbrances: {enc}; dealings waiting: {p.get('dealings_pending', 0)}.", body))
    else:
        story.append(Paragraph(f"No registered parcel at the farm's centre in the {t['registries_searched']} GeoEstate land "
                               "registries that publish a lookup.", body))
    story.append(Paragraph("Only registries on GeoEstate that publish a public lookup are searched; this is not a title search.", small))

    story.append(Paragraph("What this report is", h2))
    story.append(Paragraph("Evidence from satellites (Copernicus Sentinel-1 and -2), a weather model and land registers on the "
                           "GeoEstate platform, as at the date of issue. It is not a field inspection, a survey or a title search, "
                           "and seasons are estimates with the uncertainty shown. Farmer and crop details are as recorded by the "
                           "organization. Check that this report is genuine at " + c["verify_url"] + ".", small))
    doc.build(story)
    return buf.getvalue()


def issued_now() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.isoformat(timespec="seconds"), date.today().isoformat()
