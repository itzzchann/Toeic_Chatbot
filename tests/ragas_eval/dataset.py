"""
DATASET.PY - Golden Dataset cho RAGAS Evaluation
80 câu chọn lọc, đại diện cho 5 nhóm:
  - Nhóm 1: 22 câu Part 5 trắc nghiệm (đa dạng: thì, từ loại, giới từ, collocation)
  - Nhóm 2: 22 câu phân biệt từ vựng dễ nhầm lẫn trong TOEIC
  - Nhóm 3: 18 câu lý thuyết ngữ pháp cơ bản
  - Nhóm 4: 13 câu bắt lỗi sai / thắc mắc thường gặp
  - Nhóm 5:  5 câu out-of-scope (kiểm tra khả năng từ chối)
"""

TOEIC_GOLDEN_DATASET =[
  {
    "user_input": "Cấu trúc 'proud to V' có nghĩa là gì?",
    "references": "(S + be) + proud + to V: tự hào khi làm gì.\nVí dụ: She's proud to represent her country."
  },
  {
    "user_input": "Công thức S + be + the first/second/last + to V diễn tả gì?",
    "references": "(S + be)  + the first/second/last + to V: là người đầu tiên/thứ hai/cuối cùng làm gì"
  },
  {
    "user_input": "Khái niệm của Danh Động Từ (Gerund – V-ing Form) là gì?",
    "references": "Gerund (danh động từ) là danh từ được hình thành bằng cách thêm đuôi '-ing' vào động từ. Ví dụ: coming, building, teaching."
  },
  {
    "user_input": "Các Động Từ Khiếm Khuyết Thường Gặp gồm những từ nào?",
    "references": "Các động từ khiếm khuyết trong tiếng Anh: can, could, may, might, must, ought to, should, shall, would, will."
  },
  {
    "user_input": "Danh từ đếm được (Countable nouns) là gì?",
    "references": "Danh từ đếm được (Countable Nouns): Là những danh từ có thể “đếm” được số lượng cụ thể, có hình thức số ít và số nhiều. Ví dụ: book (books), car (cars), student (students)"
  },
  {
    "user_input": "Giới Từ Chỉ Vị Trí ngoài in, on, at còn gì không?",
    "references": "In dùng khi nói về bên trong một không gian hoặc khu vực; on dùng khi nói về vị trí trên bề mặt; at dùng để chỉ một điểm cụ thể; under dùng khi vật gì đó ở phía dưới; above và below dùng để chỉ vị trí phía trên hoặc phía dưới mà không nhất thiết phải chạm vào bề mặt."
  },
  {
    "user_input": "Câu Điều Kiện Loại 3 cấu trúc là gì?",
    "references": "If + S + V (quá khứ hoàn thành), S + would/could/might + have + Vp2. Diễn tả những giả định trái ngược với thực tế ở quá khứ."
  },
  {
    "user_input": "Kết hợp loại 3 và loại 2 (Quá khứ → Hiện tại) có công thức là gì?",
    "references": "If + S + V (quá khứ hoàn thành), S + would/could + V. Dùng để diễn tả điều kiện trái với thực tế trong quá khứ nhưng ảnh hưởng đến hiện tại."
  },
  {
    "user_input": "Cấu trúc đã từng làm gì trong quá khứ nhưng hiện tại không còn nữa được viết như thế nào?",
    "references": "(S + used)  to + V: đã từng làm gì trong quá khứ nhưng hiện tại không còn nữa. \nVí dụ: He used to play football."
  },
  {
    "user_input": "Đảo ngữ loại 3 (với had) có công thức là gì?",
    "references": "Had + S + Vp2, S + would/could/might + have + Vp2. Đảo 'had' của mệnh đề 'if' lên đầu để câu nói thêm trang trọng."
  },
  {
    "user_input": "Đảo ngữ với So... that có cấu trúc gì?",
    "references": "(So + adj) + be + (S + that) + (S + V): quá... đến mức mà..."
  },
  {
    "user_input": "Mệnh đề quan hệ xác định (Restrictive Relative Clause) là gì?",
    "references": "Mệnh đề quan hệ xác định dùng để bổ nghĩa cho danh từ đứng trước, là bộ phận quan trọng của câu, nếu bỏ đi thì mệnh đề chính không có nghĩa rõ ràng."
  },
  {
    "user_input": "Mệnh đề quan hệ không xác định (Non-restrictive Relative Clause) là gì?",
    "references": "Mệnh đề quan hệ không xác định dùng để bổ nghĩa cho danh từ đứng trước, là phần giải thích thêm. Nếu bỏ đi thì mệnh đề chính vẫn còn nghĩa rõ ràng. Không được dùng 'that' trong mệnh đề không xác định."
  },
  {
    "user_input": "Đảo ngữ với các cụm từ có NO có cấu trúc gì?",
    "references": "Cấu trúc: No/Not + N + Trợ động từ + S + Động từ. Thường dùng với các cụm như At no time, By no means, In no way, On no account, No longer, Nowhere..."
  },
  {
    "user_input": "Cấu trúc Đảo ngữ với ONLY là gì?",
    "references": "Các dạng thường gặp gồm: Only after + S + V + Trợ động từ + S + V; Only by + V-ing + Trợ động từ + S + V; Only if + S + V + Trợ động từ + S + V; Only when + S + V + Trợ động từ + S + V."
  },
  {
    "user_input": "However và Nevertheless khác nhau như thế nào?",
    "references": "However và nevertheless được sử dụng để thể hiện sự tương phản giữa hai câu riêng biệt và thường được theo sau bởi dấu phẩy."
  },
  {
    "user_input": "Cấu trúc THÌ TƯƠNG LAI HOÀN THÀNH (Future Perfect) được dùng như thế nào?",
    "references": "Thì tương lai hoàn thành dùng để diễn tả hành động hoàn thành trước một thời điểm hoặc một hành động khác trong tương lai. Dấu hiệu nhận biết: by + thời gian trong tương lai, by the end of, by the time..."
  },
  {
    "user_input": "Công thức THÌ QUÁ KHỨ ĐƠN - SIMPLE PAST VỚI ĐỘNG TỪ THƯỜNG là gì?",
    "references": "Khẳng định: S + V-ed + O. Phủ định: S + did not + V + O. Nghi vấn: Did + S + V + O?"
  },
  {
    "user_input": "Công thức THÌ QUÁ KHỨ HOÀN THÀNH - PAST PERFECT là gì?",
    "references": "Khẳng định: S + had + V3 + O. Phủ định: S + hadn't + V3 + O. Nghi vấn: Had + S + V3 + O? Dùng để diễn tả một hành động đã xảy ra và kết thúc trước một hành động khác trong quá khứ."
  },
  {
    "user_input": "Danh từ là từ dùng để chỉ gì?",
    "references": "Danh từ (Noun) là từ dùng để chỉ người, đồ vật, con vật, địa điểm, hiện tượng, khái niệm..."
  },
  {
    "user_input": "Vai trò của danh từ trong câu là những gì?",
    "references": "Danh từ có thể làm chủ ngữ, tân ngữ, bổ ngữ cho chủ ngữ, bổ ngữ cho giới từ và bổ ngữ cho tân ngữ."
  },
  {
    "user_input": "Các giới từ thông dụng là gì?",
    "references": "Một số giới từ thông dụng gồm: of, in, to, for, with, on, at, from, by, about, as, into, like."
  },
  {
    "user_input": "Có mẹo về 2 dạng cấu trúc của thể giả định là gì?",
    "references": "Câu giả định của động từ: S1 + suggest/recommend/request/ask/require/demand/insist... + S2 + (should) + V. Câu giả định của tính từ: It + be + crucial/vital/essential/mandatory/necessary + (that) + S + (should) + V."
  },
  {
    "user_input": "Mệnh đề quan hệ nối tiếp là gì?",
    "references": "Mệnh đề quan hệ nối tiếp dùng để giải thích cả một câu, chỉ dùng đại từ quan hệ 'which' và dùng dấu phẩy để tách hai mệnh đề. Mệnh đề này luôn đứng ở cuối câu."
  },
  {
    "user_input": "Câu bị động là câu gì?",
    "references": "Câu bị động là câu mà chủ ngữ chính là đối tượng chịu tác động của hành động, thay vì là người thực hiện hành động. Trong tiếng Việt, chúng ta có thể dùng 'bị/được' để diễn tả."
  },
  {
    "user_input": "Dấu hiệu của Present Simple (Hiện tại đơn) là gì?",
    "references": "Vị trí trạng từ tần suất thường gặp trong thì hiện tại đơn gồm: always, usually, often, sometimes, seldom, rarely, never."
  },
  {
    "user_input": "Sự khác nhau cơ bản giữa thì hiện tại đơn và thì hiện tại tiếp diễn là gì?",
    "references": "Thì hiện tại đơn diễn tả thói quen, năng lực hoặc kế hoạch, thời khóa biểu đã sắp xếp trước. Thì hiện tại tiếp diễn diễn tả hành động đang diễn ra tại thời điểm nói. Công thức hiện tại tiếp diễn: S + be (am/is/are) + V-ing + O."
  },
  {
    "user_input": "Quy tắc dùng 'a' trước danh từ như thế nào?",
    "references": "Dùng 'a' trước danh từ bắt đầu bằng một phụ âm. Được dùng trước một danh từ không xác định về vị trí, tính chất hoặc được nhắc đến lần đầu trong câu."
  },
  {
    "user_input": "Cụm từ 'responsible for' nghĩa là gì?",
    "references": "Cụm 'responsible for' có nghĩa là chịu trách nhiệm cho."
  },
  {
    "user_input": "Cụm từ 'Propose to' nghĩa là gì?",
    "references": "Propose to do something: có ý định làm gì. Propose doing something: đề nghị làm gì."
  },
  {
    "user_input": "Phân biệt Liên từ và Giới từ (Conjunction and Preposition) như thế nào?",
    "references": "Liên từ dùng để liên kết hai mệnh đề. Giới từ dùng để liên kết danh từ với danh từ hoặc danh từ với mệnh đề. Nếu phía sau là mệnh đề thì dùng liên từ, nếu là danh từ hoặc cụm danh từ thì dùng giới từ."
  },
  {
    "user_input": "2 từ after và before được dùng trong quá khứ hoàn thành (Past Perfect) như thế nào?",
    "references": "Mệnh đề có after và before có thể đứng ở đầu hoặc cuối câu. Sau after thường dùng thì quá khứ hoàn thành (Past Perfect), còn sau before thường dùng thì quá khứ đơn (Simple Past)."
  },
  {
    "user_input": "Trong trường hợp có giới từ đứng trước đại từ quan hệ thì làm gì?",
    "references": "Trong trường hợp có giới từ đứng trước đại từ quan hệ, có thể đảo giới từ ra cuối mệnh đề quan hệ. Ví dụ: I saw the boy to whom you talked yesterday → I saw the boy you talked to yesterday."
  },
  {
    "user_input": "That là đại từ quan hệ thay thế được cho gì?",
    "references": "That là đại từ quan hệ chỉ cả người lẫn vật, có thể được dùng thay cho who, whom, which trong mệnh đề quan hệ xác định."
  },
  {
    "user_input": "Rút gọn mệnh đề quan hệ bằng cách sử dụng cụm danh từ như thế nào?",
    "references": "Mệnh đề quan hệ không xác định có thể được rút gọn bằng cách bỏ đại từ quan hệ và động từ be, giữ lại danh từ, cụm danh từ hoặc cụm giới từ phía sau."
  },
  {
    "user_input": "Phân biệt cách dùng 'stop + V-ing' và 'stop + to V'?",
    "references": "Stop + V-ing: dừng việc đang làm lại. Stop + to V: dừng lại để chuyển sang thực hiện một việc khác."
  },
  {
    "user_input": "Thay thế 'if' trong câu điều kiện có thật bằng những từ nào?",
    "references": "Trong câu điều kiện có thật, 'if' có thể được thay bằng: when, in case, as long as, so long as, provided that, providing that, only if, on the condition (that)."
  },
  {
    "user_input": "Phrasal verb 'look forward to' có nghĩa là gì và dùng như thế nào?",
    "references": "Look forward to + N/V-ing: trông đợi, mong chờ điều gì. Ví dụ: I'm looking forward to receiving your email."
  },
  {
    "user_input": "Cụm từ 'Look on the bright side' có nghĩa là gì?",
    "references": "Look on the bright side: lạc quan, nhìn vào mặt tích cực của vấn đề."
  },
  {
    "user_input": "Phrasal verb 'look into' có nghĩa là gì?",
    "references": "Look into: điều tra, nghiên cứu hoặc xem xét một vấn đề."
  },
  {
    "user_input": "Phrasal verb 'look up' có nghĩa là gì?",
    "references": "Look up: tra cứu thông tin trong sách, từ điển hoặc tài liệu tham khảo."
  },
  {
    "user_input": "Cụm từ 'instead of' có nghĩa là gì?",
    "references": "Instead of là giới từ mang nghĩa 'thay vì'."
  },
  {
    "user_input": "Khi nào dùng cấu trúc O + V-ing sau các động từ tri giác?",
    "references": "Các động từ như hear, sound, smell, taste, feel, watch, notice, see, listen, find + O + V-ing dùng để chỉ khoảnh khắc hành động đang diễn ra."
  },
  {
    "user_input": "Danh từ tận cùng bằng F hoặc FE chuyển sang số nhiều như thế nào?",
    "references": "Danh từ tận cùng bằng F hoặc FE thường đổi thành ves khi chuyển sang số nhiều. Ví dụ: leaf → leaves, knife → knives."
  },
  {
    "user_input": "Những danh từ tận cùng bằng F hoặc FE nào là ngoại lệ khi chuyển sang số nhiều?",
    "references": "Một số ngoại lệ chỉ thêm s: roofs, gulfs, cliffs, reefs, proofs, chiefs, safes, dwarfs, turfs, griefs, beliefs."
  },
  {
    "user_input": "Thay đổi trạng từ thời Gian/địa Điểm (Time/Place Adverbs) trong câu tường thuật như thế nào?",
    "references": "Một số thay đổi thường gặp trong câu tường thuật: now → then, today → that day, yesterday → the day before/the previous day, tomorrow → the day after/the next day, this week → that week, last month → the month before/the previous month, next year → the following year, here → there, this → that, these → those."
  },
  {
    "user_input": "Interjections (Thán từ) là gì?",
    "references": "Interjections (thán từ) là những từ dùng để bộc lộ cảm xúc bất ngờ. Ví dụ: Wow!, Oh!, Oops!, Hey!"
  },
  {
    "user_input": "Cấu trúc THÌ TƯƠNG LAI đơn - SIMPLE FUTURE là gì?",
    "references": "Khẳng định: S + shall/will + V (infinitive) + O. Phủ định: S + shall/will + not + V (infinitive) + O. Nghi vấn: Shall/Will + S + V (infinitive) + O?"
  },
  {
    "user_input": "Câu giả định dùng với it + to be + time là gì?",
    "references": "It's time (for sb) to do something: đã đến lúc ai phải làm gì (thời gian vừa vặn, không mang tính giả định). Ví dụ: It's time for me to get to the airport. It's time / It's high time / It's about time + S + V (quá khứ đơn): đã đến lúc mà (mang tính giả định, hàm ý đã trễ). Ví dụ: It's time I got to the airport. It's high time the city government did something to stop the traffic jam."
  },
  {
    "user_input": "Không Dùng Mạo Từ khi nào?",
    "references": "Không dùng mạo từ (Zero Article) trong các trường hợp: trước danh từ số nhiều mang nghĩa chung chung (Cars are expensive nowadays), trước danh từ không đếm được mang nghĩa chung chung (Information is crucial), trước tên riêng như người, thành phố, quốc gia, ngôn ngữ (I live in Vietnam), và trong một số cụm cố định như go to school, at home, have breakfast."
  }
]