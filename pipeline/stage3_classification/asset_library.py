"""
Static asset library — placeholder mesh IDs + CLIP-compatible label strings.
All IDs use the naming convention that Stage 5 (Assembly) will expect.
"""

HAIR_ASSETS = [
    {"id": "hair_bald_001",          "label": "bald or shaved head with no hair"},
    {"id": "hair_buzzcut_001",       "label": "buzz cut very short hair"},
    {"id": "hair_pixie_001",         "label": "pixie cut short feminine hair"},
    {"id": "hair_bob_001",           "label": "bob cut chin-length hair"},
    {"id": "hair_short_straight_001","label": "short straight hair"},
    {"id": "hair_short_wavy_001",    "label": "short wavy hair"},
    {"id": "hair_short_curly_001",   "label": "short curly hair"},
    {"id": "hair_undercut_001",      "label": "undercut fade hairstyle"},
    {"id": "hair_waves_001",         "label": "360 waves short textured hair"},
    {"id": "hair_mohawk_001",        "label": "mohawk or faux-hawk"},
    {"id": "hair_medium_straight_001","label": "medium length straight hair reaching shoulders"},
    {"id": "hair_medium_wavy_001",   "label": "medium length wavy hair reaching shoulders"},
    {"id": "hair_medium_curly_001",  "label": "medium length curly hair reaching shoulders"},
    {"id": "hair_bangs_straight_001","label": "straight fringe or blunt bangs"},
    {"id": "hair_bangs_side_001",    "label": "side-swept bangs or curtain bangs"},
    {"id": "hair_long_straight_001", "label": "long straight hair past shoulders"},
    {"id": "hair_long_wavy_001",     "label": "long wavy hair past shoulders"},
    {"id": "hair_long_curly_001",    "label": "long curly hair past shoulders"},
    {"id": "hair_ponytail_high_001", "label": "high ponytail"},
    {"id": "hair_ponytail_low_001",  "label": "low ponytail"},
    {"id": "hair_bun_top_001",       "label": "top knot or messy bun on top of head"},
    {"id": "hair_bun_low_001",       "label": "low bun at nape of neck"},
    {"id": "hair_braid_single_001",  "label": "single braid or plait"},
    {"id": "hair_braids_double_001", "label": "two braids or pigtails"},
    {"id": "hair_braids_box_001",    "label": "box braids or cornrows"},
    {"id": "hair_afro_001",          "label": "afro natural hair"},
    {"id": "hair_coily_001",         "label": "tight coils or 4c natural hair"},
    {"id": "hair_dreadlocks_001",    "label": "dreadlocks or locs"},
    {"id": "hair_twists_001",        "label": "two-strand twists or Bantu knots"},
    {"id": "hair_messy_001",         "label": "messy tousled or bedhead hair"},
]

FACE_ASSETS = [
    {"id": "face_oval_001",     "label": "oval face shape balanced forehead and chin"},
    {"id": "face_round_001",    "label": "round face shape full cheeks equal width and height"},
    {"id": "face_square_001",   "label": "square face shape strong angular jaw wide forehead"},
    {"id": "face_heart_001",    "label": "heart-shaped face wide forehead narrow pointed chin"},
    {"id": "face_diamond_001",  "label": "diamond face shape narrow forehead and chin wide cheekbones"},
    {"id": "face_oblong_001",   "label": "oblong or rectangular face shape long narrow"},
    {"id": "face_triangle_001", "label": "pear or triangle face shape narrow forehead wide jaw"},
    {"id": "face_wide_001",     "label": "wide or broad face prominent cheekbones"},
]

# ~10 garment types × 5 fit presets = ~50 meshes
# The ID suffix _S/_R/_L/_B/_O encodes fit: Slim/Regular/Loose/Baggy/Oversized
_FITS = ["slim", "regular", "loose", "baggy", "oversized"]

GARMENT_ASSETS = [
    # ── Tops ──────────────────────────────────────────────────────────────────
    {"id": "top_tshirt_001",      "label": "t-shirt",                      "category": "top"},
    {"id": "top_polo_001",        "label": "polo shirt",                   "category": "top"},
    {"id": "top_buttondown_001",  "label": "button-down or dress shirt",   "category": "top"},
    {"id": "top_hoodie_001",      "label": "hoodie or sweatshirt",         "category": "top"},
    {"id": "top_croptop_001",     "label": "crop top or midriff top",      "category": "top"},
    {"id": "top_tanktop_001",     "label": "tank top or sleeveless shirt", "category": "top"},
    {"id": "top_longsleeve_001",  "label": "long sleeve shirt or henley",  "category": "top"},
    {"id": "top_turtleneck_001",  "label": "turtleneck or mock-neck top",  "category": "top"},
    {"id": "top_sweater_001",     "label": "sweater or knit top",          "category": "top"},
    {"id": "top_athletic_001",    "label": "athletic jersey or sports top","category": "top"},
    # ── Outerwear ─────────────────────────────────────────────────────────────
    {"id": "outer_blazer_001",    "label": "blazer or suit jacket",        "category": "outerwear"},
    {"id": "outer_bomber_001",    "label": "bomber or varsity jacket",     "category": "outerwear"},
    {"id": "outer_denim_001",     "label": "denim or jean jacket",         "category": "outerwear"},
    {"id": "outer_leather_001",   "label": "leather or moto jacket",       "category": "outerwear"},
    {"id": "outer_puffer_001",    "label": "puffer or down jacket",        "category": "outerwear"},
    {"id": "outer_windbreaker_001","label": "windbreaker or zip-up jacket","category": "outerwear"},
    {"id": "outer_cardigan_001",  "label": "cardigan or open-front knit",  "category": "outerwear"},
    {"id": "outer_coat_001",      "label": "coat or overcoat",             "category": "outerwear"},
    {"id": "outer_vest_001",      "label": "vest or gilet",                "category": "outerwear"},
    # ── Bottoms ───────────────────────────────────────────────────────────────
    {"id": "bottom_jeans_001",    "label": "jeans or denim pants",         "category": "bottom"},
    {"id": "bottom_chinos_001",   "label": "chinos khakis or dress pants", "category": "bottom"},
    {"id": "bottom_joggers_001",  "label": "joggers or sweatpants",        "category": "bottom"},
    {"id": "bottom_cargo_001",    "label": "cargo pants with pockets",     "category": "bottom"},
    {"id": "bottom_shorts_001",   "label": "shorts or bermuda shorts",     "category": "bottom"},
    {"id": "bottom_leggings_001", "label": "leggings or athletic tights",  "category": "bottom"},
    {"id": "bottom_skirt_mini_001","label": "mini skirt above the knee",   "category": "bottom"},
    {"id": "bottom_skirt_midi_001","label": "midi skirt knee to calf",     "category": "bottom"},
    {"id": "bottom_skirt_maxi_001","label": "maxi skirt ankle length",     "category": "bottom"},
    # ── Dresses ───────────────────────────────────────────────────────────────
    {"id": "dress_casual_001",    "label": "casual everyday dress",        "category": "dress"},
    {"id": "dress_bodycon_001",   "label": "bodycon or fitted dress",      "category": "dress"},
    {"id": "dress_sundress_001",  "label": "sundress or summer dress",     "category": "dress"},
    {"id": "dress_formal_001",    "label": "formal or evening gown",       "category": "dress"},
    # ── Footwear ──────────────────────────────────────────────────────────────
    {"id": "shoes_sneakers_001",  "label": "sneakers or athletic shoes",   "category": "footwear"},
    {"id": "shoes_boots_001",     "label": "boots or ankle boots",         "category": "footwear"},
    {"id": "shoes_loafers_001",   "label": "loafers or casual shoes",      "category": "footwear"},
    {"id": "shoes_heels_001",     "label": "heels or pumps",               "category": "footwear"},
    {"id": "shoes_sandals_001",   "label": "sandals or flip flops",        "category": "footwear"},
    {"id": "shoes_dress_001",     "label": "dress shoes or oxfords",       "category": "footwear"},
    {"id": "shoes_platform_001",  "label": "platform or chunky shoes",     "category": "footwear"},
]

ACCESSORY_ASSETS = [
    # ── Eyewear ───────────────────────────────────────────────────────────────
    {"id": "glasses_round_001",       "label": "round glasses",                  "category": "eyewear"},
    {"id": "glasses_rectangle_001",   "label": "rectangular glasses",            "category": "eyewear"},
    {"id": "glasses_aviator_001",     "label": "aviator sunglasses",             "category": "eyewear"},
    {"id": "glasses_cat_eye_001",     "label": "cat-eye glasses",                "category": "eyewear"},
    {"id": "glasses_oversized_001",   "label": "oversized sunglasses",           "category": "eyewear"},
    # ── Hats ──────────────────────────────────────────────────────────────────
    {"id": "hat_baseball_001",        "label": "baseball cap or snapback",       "category": "hat"},
    {"id": "hat_beanie_001",          "label": "beanie or knit hat",             "category": "hat"},
    {"id": "hat_fedora_001",          "label": "fedora or wide-brim hat",        "category": "hat"},
    {"id": "hat_bucket_001",          "label": "bucket hat",                     "category": "hat"},
    {"id": "hat_cowboy_001",          "label": "cowboy or western hat",          "category": "hat"},
    # ── Bags ──────────────────────────────────────────────────────────────────
    {"id": "bag_backpack_001",        "label": "backpack",                       "category": "bag"},
    {"id": "bag_shoulder_001",        "label": "shoulder bag or purse",          "category": "bag"},
    {"id": "bag_tote_001",            "label": "tote bag",                       "category": "bag"},
    {"id": "bag_crossbody_001",       "label": "crossbody or sling bag",         "category": "bag"},
    {"id": "bag_fanny_001",           "label": "fanny pack or belt bag",         "category": "bag"},
    # ── Watches ───────────────────────────────────────────────────────────────
    {"id": "watch_sport_001",         "label": "sport watch or smartwatch",      "category": "watch"},
    {"id": "watch_casual_001",        "label": "casual analog watch",            "category": "watch"},
    {"id": "watch_dress_001",         "label": "dress watch",                    "category": "watch"},
    # ── Headphones ────────────────────────────────────────────────────────────
    {"id": "headphones_over_001",     "label": "over-ear headphones",            "category": "headphones"},
    {"id": "headphones_ear_001",      "label": "earbuds or in-ear headphones",   "category": "headphones"},
    # ── Jewelry ───────────────────────────────────────────────────────────────
    {"id": "necklace_chain_001",      "label": "chain necklace",                 "category": "necklace"},
    {"id": "necklace_pendant_001",    "label": "pendant necklace",               "category": "necklace"},
    {"id": "earrings_stud_001",       "label": "stud earrings",                  "category": "earrings"},
    {"id": "earrings_hoop_001",       "label": "hoop earrings",                  "category": "earrings"},
    {"id": "earrings_dangle_001",     "label": "dangle or drop earrings",        "category": "earrings"},
    {"id": "bracelet_001",            "label": "bracelet or wristband",          "category": "bracelet"},
    # ── Other ─────────────────────────────────────────────────────────────────
    {"id": "belt_001",                "label": "belt",                           "category": "belt"},
    {"id": "scarf_001",               "label": "scarf or neck wrap",             "category": "scarf"},
    {"id": "gloves_001",              "label": "gloves",                         "category": "gloves"},
]

GARMENT_FIT_OPTIONS = ["slim", "regular", "loose", "baggy", "oversized"]

ACCESSORY_DETECTION_THRESHOLD = 0.38
