"""Câu mô phỏng từ mô tả V1, cần rà soát bệnh học và đối chiếu ảnh.

Mỗi lớp có 6 họ câu chi tiết và 6 họ câu một phần.
4 họ đầu dành cho Train, họ thứ 5 cho Val, họ thứ 6 cho Test.
Một họ và mọi câu ghép từ họ đó chỉ thuộc một split.
Không dùng tên bệnh, label, mã lớp, mùa hay điều kiện thời tiết trong text.
"""
from itertools import product
from pipeline_utils import SPLITS, clean_vietnamese_text

# Họ câu theo thứ tự cố định: train, train, train, train, val, test.
# 'partial' có ít dấu hiệu hơn; không cố ý gán triệu chứng sai cho lớp.
PROFILES = {
    "Apple___Apple_scab": {
        "detailed": [
            "lá có những đốm sẫm màu với bề mặt nhám",
            "trên phiến lá xuất hiện vết đen sần và hơi xoăn",
            "các đốm tối trên lá có bề mặt không nhẵn",
            "lá có vết thâm nhám và phần mép hơi cuộn",
            "bề mặt lá mang các mảng sẫm sần sùi",
            "những vết tối ráp xuất hiện trên phần lá bị biến dạng",
        ],
        "partial": ["lá có vài đốm sẫm", "phiến lá hơi xoăn", "trên lá có mảng thâm", "mép lá hơi cuộn", "lá xuất hiện vùng tối màu", "lá có chỗ bề mặt không còn nhẵn"],
    },
    "Apple___Black_rot": {
        "detailed": [
            "lá có đốm nâu với viền tím đỏ",
            "ở giữa vết tròn là màu nâu và quanh vết có màu tía",
            "những đốm tròn trên lá có tâm nâu viền đỏ",
            "phiến lá mang vết nâu được bao bởi quầng tím",
            "vết trên lá có phần giữa nâu nhạt và rìa đỏ tía",
            "các vùng tổn thương dạng tròn có tâm nâu và mép tím đỏ",
        ],
        "partial": ["lá có vài đốm nâu", "trên phiến lá có vết tròn", "lá xuất hiện chấm sẫm", "một vùng lá bị thâm", "lá mang các vệt nâu nhỏ", "phiến lá có nhiều chấm đổi màu"],
    },
    "Apple___Cedar_apple_rust": {
        "detailed": [
            "mặt trên lá có những đốm vàng cam rõ nét",
            "trên lá xuất hiện các chấm tròn màu cam sáng",
            "phiến lá có đốm vàng pha cam với ranh giới rõ",
            "những vùng tròn màu cam nổi bật trên nền lá",
            "mặt lá có vết vàng ánh cam dễ nhận thấy",
            "các đốm tròn trên bề mặt lá chuyển sang sắc cam vàng",
        ],
        "partial": ["lá có vài đốm vàng", "mặt lá có chấm sáng màu", "trên lá có vùng ngả vàng", "phiến lá đổi màu từng chỗ", "lá xuất hiện vết màu sáng", "một số vùng của lá có màu khác thường"],
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "detailed": [
            "lá có vệt xám hẹp chạy theo gân",
            "các vết xám trên lá có dạng chữ nhật kéo dài",
            "những sọc màu xám nằm dọc đường gân lá",
            "phiến lá có vết hẹp màu tro với cạnh khá thẳng",
            "các mảng xám thuôn hẹp phân bố dọc gân",
            "vùng đổi màu trên lá tạo thành dải hình chữ nhật xám",
        ],
        "partial": ["lá có vệt xám", "phiến lá có vài sọc nhạt", "trên lá có vùng khô", "lá có các mảng đổi màu", "trên phiến lá xuất hiện vệt nhạt dài", "lá có một số vùng màu xám nâu"],
    },
    "Corn_(maize)___Common_rust_": {
        "detailed": [
            "mặt lá có nốt nhỏ nâu đỏ với bột mịn",
            "các chấm nổi màu cam nâu xuất hiện trên lá",
            "phiến lá có mụn màu đỏ nâu trông như bụi gỉ",
            "lá mang nhiều nốt ráp nhỏ màu nâu cam",
            "bề mặt lá có đám nốt màu đỏ sẫm dạng bột",
            "những điểm nhô lên trên lá có màu nâu đỏ và lớp bột",
        ],
        "partial": ["lá có chấm nâu", "trên lá có các nốt nhỏ", "bề mặt lá hơi ráp", "phiến lá có đốm sẫm", "mặt lá xuất hiện chấm đỏ nhạt", "lá có các vùng nhỏ nhô lên"],
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "detailed": [
            "lá có mảng nâu xám dài với hai đầu thuôn",
            "vết khô trên lá kéo dài như hình điếu xì gà",
            "những mảng cháy lớn hình thon nằm trên phiến lá",
            "lá mang vết nâu nhạt dài và thu hẹp ở hai đầu",
            "các vùng khô xám nâu trên lá có dạng thoi dài",
            "phiến lá có mảng hoại tử thuôn dài màu nâu xám",
        ],
        "partial": ["lá có mảng khô", "trên lá có vết nâu dài", "phiến lá bị thâm từng vùng", "một phần lá có màu xám", "lá xuất hiện vùng nâu khá lớn", "có mảng đổi màu kéo dài trên lá"],
    },
    "Tomato___Bacterial_spot": {
        "detailed": [
            "lá có chấm đen nhỏ với viền vàng",
            "những đốm nâu sẫm trên lá được bao quanh bởi quầng vàng",
            "phiến lá có nhiều chấm tối và mép hơi nhăn",
            "lá có vết nhỏ sậm màu kèm vùng vàng quanh vết",
            "các chấm thâm nhỏ trên lá có rìa vàng nhạt",
            "trên phiến lá xuất hiện đốm tối li ti với quầng sáng vàng",
        ],
        "partial": ["lá có vài chấm đen", "mép lá hơi nhăn", "trên lá có đốm nhỏ", "phiến lá có chỗ ngả vàng", "lá xuất hiện các điểm thâm nhỏ", "có những chấm đổi màu rải trên lá"],
    },
    "Tomato___Early_blight": {
        "detailed": [
            "lá có đốm nâu với nhiều vòng đồng tâm",
            "vết nâu trên phiến lá có vân tròn và rìa vàng",
            "những đốm sẫm trên lá chứa vòng tròn lồng nhau",
            "lá có vùng thâm nâu với hoa văn dạng bia tròn",
            "các vết nâu mang đường vòng xếp lớp và quầng vàng",
            "trên lá có đốm với các vòng tròn cùng tâm màu nâu",
        ],
        "partial": ["lá có đốm nâu", "phiến lá bị vàng từng vùng", "lá có vết thâm tròn", "một phần lá đổi sang nâu", "trên lá xuất hiện mảng nâu sẫm", "lá có vùng vàng quanh chỗ thâm"],
    },
    "Tomato___Late_blight": {
        "detailed": [
            "lá có mảng nâu xám giống úng nước và mốc trắng ở mặt dưới",
            "vùng thâm trên lá trông ướt và mặt dưới có lớp trắng",
            "phiến lá có vết nâu không đều kèm lớp mốc nhạt",
            "lá xuất hiện mảng xám úng với tơ trắng phía dưới",
            "mặt dưới vùng nâu xám ướt trên lá có lớp mốc màu trắng",
            "lá mang tổn thương sẫm giống ngấm nước và lớp tơ trắng bên dưới",
        ],
        "partial": ["lá có mảng thâm", "phiến lá có vùng nâu xám", "trên lá có vết trông ướt", "lá bị thâm từng mảng", "một phần lá chuyển màu sẫm", "lá có vùng xám nâu không đều"],
    },
}

# Cùng ngân hàng cho mọi lớp healthy để tránh tên cây quyết định đáp án.
HEALTHY = {
    "detailed": [
        "lá xanh đều và không thấy đốm lạ",
        "phiến lá còn nguyên với bề mặt nhẵn màu xanh",
        "lá có màu xanh ổn định và không thấy vùng khô",
        "bề mặt lá đều màu không có vết thâm rõ",
        "lá giữ sắc xanh và chưa thấy dấu tổn thương",
        "phiến lá xanh nguyên vẹn không xuất hiện mảng đổi màu",
    ],
    "partial": ["lá vẫn xanh", "mặt lá khá đều màu", "phiến lá còn nguyên", "lá chưa có gì nổi bật", "bề mặt lá trông bình thường", "chưa thấy thay đổi rõ trên phiến lá"],
}
for label in ("Apple___healthy", "Corn_(maize)___healthy", "Tomato___healthy"):
    PROFILES[label] = HEALTHY

# Các mô tả thiếu thông tin được chia riêng và dùng chung cho cả 12 lớp.
GENERIC = {
    "train": ["tôi chưa ghi lại chi tiết của lá", "lá cần được quan sát kỹ hơn", "tôi chỉ mới chụp lá để kiểm tra", "chưa đủ thông tin để mô tả biểu hiện", "tôi không chắc cần chú ý phần nào của lá", "ảnh lá là thông tin chính tôi có lúc này"],
    "val": ["tôi chưa biết diễn tả tình trạng phiến lá", "chưa quan sát được các chi tiết nhỏ trên lá"],
    "test": ["hiện tôi chỉ có hình lá và ít thông tin", "cần xem thêm ảnh vì tôi mô tả chưa rõ"],
}

# Dùng cùng cách nói ở cả ba split để tránh split gắn với một văn phong.
PREFIXES = ["Quan sát thấy", "Tôi thấy", "Nhìn vào lá thì", "Theo quan sát của tôi", "Kiểm tra lá thấy", "Phần lá được chụp cho thấy"]
SUFFIXES = [".", ", đây là phần tôi quan sát được.", ", tôi muốn kiểm tra tình trạng này."]


def make_text_bank():
    bank = {split: {} for split in SPLITS}
    family_indices = {"train": range(4), "val": [4], "test": [5]}
    for split in SPLITS:
        for label, profile in sorted(PROFILES.items()):
            levels = {}
            for level in ("detailed", "partial"):
                entries = []
                for i in family_indices[split]:
                    core = profile[level][i]
                    # Câu healthy dùng chung có cùng family id, cả ở khác cây.
                    family = f"{level}:{clean_vietnamese_text(core)}"
                    for prefix, suffix in product(PREFIXES, SUFFIXES):
                        entries.append({"description": f"{prefix} {core}{suffix}", "family_id": family})
                levels[level] = entries
            levels["generic"] = [
                {"description": f"{prefix} {core}{suffix}", "family_id": f"generic:{clean_vietnamese_text(core)}"}
                for core, prefix, suffix in product(GENERIC[split], PREFIXES, SUFFIXES)
            ]
            bank[split][label] = levels
    # Kiểm tra tất cả câu có thể sinh, kể cả câu chưa được chọn vào dataset.
    pools = {}
    for split in SPLITS:
        pools[split] = {clean_vietnamese_text(e["description"])
                        for levels in bank[split].values() for entries in levels.values() for e in entries}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            if pools[a] & pools[b]:
                raise ValueError(f"Ngân hàng câu trùng sau làm sạch: {a}/{b}")
    return bank
