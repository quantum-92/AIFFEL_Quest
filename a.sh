# 대용량 파일을 모든 커밋에서 제거
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch Flutter_quest/MainQuest/stock_prediction/feature_list.npy" \
  --prune-empty --tag-name-filter cat -- --all

# Git 가비지 컬렉션 실행 및 리포지토리 최적화
git gc --prune=now
git repack -a -d

