# Bệnh hiếm: 1% dân số mắc.
# Xét nghiệm: đúng 99% với người bệnh, báo sai 5% với người khỏe.
# Xét nghiệm dương tính -> xác suất THỰC SỰ mắc bệnh là bao nhiêu?

P_benh          = 0.01
P_duong_neu_benh = 0.99
P_duong_neu_khoe = 0.05

P_duong = P_duong_neu_benh * P_benh + P_duong_neu_khoe * (1 - P_benh)
P_benh_neu_duong = (P_duong_neu_benh * P_benh) / P_duong

print(f"P(bệnh | dương tính) = {P_benh_neu_duong:.2%}")   # 16.67%  <- trực giác sai hoàn toàn!