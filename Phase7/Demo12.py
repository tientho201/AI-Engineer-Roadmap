"""
3 kiến trúc voice agent, đánh đổi độ trễ vs kiểm soát:

1. Cascaded (STT -> LLM -> TTS)
   Độ trễ ~1.5-3s. Kiểm soát tốt nhất, debug dễ, thay từng phần được.

2. Speech-to-Speech (Realtime API)
   Độ trễ ~300-800ms. Tự nhiên nhất nhưng khó kiểm soát nội dung.

3. Hybrid: realtime cho hội thoại + cascaded cho tác vụ cần độ chính xác
"""
import asyncio

async def cascaded_voice_agent(audio_stream):
    # 1) VAD: phát hiện người dùng ngừng nói
    #    Đây là phần KHÓ NHẤT — cắt sớm thì ngắt lời, cắt muộn thì cảm giác chậm
    async for utterance in vad_segment(audio_stream, silence_ms=500):

        # 2) STT streaming — bắt đầu transcribe NGAY, không chờ nói xong
        text = await transcribe_streaming(utterance)

        # 3) LLM streaming
        buffer = ""
        async for delta in llm_stream(text):
            buffer += delta
            # 4) TTS theo từng CÂU — không chờ LLM trả hết
            #    Mẹo then chốt để giảm độ trễ cảm nhận
            if any(buffer.rstrip().endswith(p) for p in ".!?。！？"):
                await tts_play(buffer)
                buffer = ""
        if buffer:
            await tts_play(buffer)


# NGÂN SÁCH ĐỘ TRỄ cho cảm giác "tự nhiên" (tổng < 800ms):
#   VAD phát hiện kết thúc : 200-500 ms
#   STT                     : 100-300 ms
#   LLM (token đầu tiên)    : 200-500 ms   <- prompt caching giúp nhiều
#   TTS (âm đầu tiên)       : 100-200 ms
#
# XỬ LÝ NGẮT LỜI (barge-in): người dùng nói đè -> phải DỪNG TTS ngay
# và hủy request LLM đang chạy. Không làm được điều này thì agent
# nghe rất "máy móc".