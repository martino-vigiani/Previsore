"""Mappa nazionale -> confederazione (UEFA/CONMEBOL/CONCACAF/CAF/AFC/OFC).

Serve a stimare l'offset di forza TRA confederazioni: le squadre giocano per lo
piu dentro la propria confederazione (qualificazioni), quindi il livello relativo
fra confederazioni e mal vincolato dai soli attacco/difesa individuali. Le partite
cross-confederation (Mondiale, spareggi intercont., amichevoli) lo identificano.

Lookup robusto: minuscolo + senza accenti. Nomi nella grafia martj42.
"""
from __future__ import annotations

import unicodedata

CONFEDERATIONS = ("UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC")

_MEMBERS = {
    "CONMEBOL": [
        "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
        "Paraguay", "Peru", "Uruguay", "Venezuela",
    ],
    "UEFA": [
        "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium",
        "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
        "Denmark", "England", "Estonia", "Faroe Islands", "Finland", "France", "Georgia",
        "Germany", "Gibraltar", "Greece", "Hungary", "Iceland", "Israel", "Italy",
        "Kazakhstan", "Kosovo", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg",
        "Malta", "Moldova", "Montenegro", "Netherlands", "North Macedonia",
        "Northern Ireland", "Norway", "Poland", "Portugal", "Republic of Ireland",
        "Romania", "Russia", "San Marino", "Scotland", "Serbia", "Slovakia", "Slovenia",
        "Spain", "Sweden", "Switzerland", "Turkey", "Ukraine", "Wales",
    ],
    "CONCACAF": [
        "Anguilla", "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados", "Belize",
        "Bermuda", "British Virgin Islands", "Canada", "Cayman Islands", "Costa Rica",
        "Cuba", "Curaçao", "Dominica", "Dominican Republic", "El Salvador", "Grenada",
        "Guadeloupe", "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica", "Martinique",
        "Mexico", "Montserrat", "Nicaragua", "Panama", "Puerto Rico", "Saint Kitts and Nevis",
        "Saint Lucia", "Saint Vincent and the Grenadines", "Sint Maarten", "Suriname",
        "Trinidad and Tobago", "Turks and Caicos Islands", "United States",
        "US Virgin Islands", "French Guiana",
    ],
    "CAF": [
        "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon",
        "Cape Verde", "Central African Republic", "Chad", "Comoros", "Congo", "DR Congo",
        "Ivory Coast", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini",
        "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Kenya",
        "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania",
        "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda",
        "São Tomé and Príncipe", "Senegal", "Seychelles", "Sierra Leone", "Somalia",
        "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda",
        "Zambia", "Zimbabwe", "Swaziland",
    ],
    "AFC": [
        "Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan", "Brunei",
        "Cambodia", "China PR", "China", "Chinese Taipei", "Guam", "Hong Kong", "India",
        "Indonesia", "Iran", "Iraq", "Japan", "Jordan", "Kuwait", "Kyrgyzstan", "Laos",
        "Lebanon", "Macau", "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal",
        "North Korea", "Oman", "Pakistan", "Palestine", "Philippines", "Qatar",
        "Saudi Arabia", "Singapore", "South Korea", "Sri Lanka", "Syria", "Tajikistan",
        "Thailand", "Timor-Leste", "Turkmenistan", "United Arab Emirates", "Uzbekistan",
        "Vietnam", "Yemen",
    ],
    "OFC": [
        "American Samoa", "Cook Islands", "Fiji", "New Caledonia", "New Zealand",
        "Papua New Guinea", "Samoa", "Solomon Islands", "Tahiti", "Tonga", "Vanuatu",
    ],
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).casefold().strip()


_LOOKUP = {_norm(team): conf for conf, teams in _MEMBERS.items() for team in teams}


def conf_of(team: str):
    """Confederazione della nazionale, o None se sconosciuta (effetto neutro)."""
    return _LOOKUP.get(_norm(team))
