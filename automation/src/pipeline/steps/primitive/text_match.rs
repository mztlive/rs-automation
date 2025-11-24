use unicode_normalization::UnicodeNormalization;

pub const DEFAULT_MIN_SIMILARITY: f32 = 0.6;

/// 判断 OCR 文本与目标词是否“足够相似”，使用简单的 Levenshtein 相似度。
pub fn has_similar(text: &str, pattern: &str, case_sensitive: bool) -> bool {
    has_similar_with_threshold(text, pattern, case_sensitive, DEFAULT_MIN_SIMILARITY)
}

pub fn has_similar_with_threshold(
    text: &str,
    pattern: &str,
    case_sensitive: bool,
    min_similarity: f32,
) -> bool {
    if text.is_empty() || pattern.is_empty() {
        return false;
    }

    let normalized_text = normalize(text, case_sensitive);
    let normalized_pattern = normalize(pattern, case_sensitive);

    // 先做包含判断，匹配“子串完全出现”场景（中文短词时更宽松）。
    if normalized_text.contains(&normalized_pattern)
        || normalized_pattern.contains(&normalized_text)
    {
        return true;
    }

    composite_similarity(&normalized_text, &normalized_pattern) >= min_similarity
}

/// 取 Levenshtein 相似度与最长公共子序列（LCS）相似度的最大值，以兼顾
/// - 编辑距离（常规错别字、OCR 噪声）
/// - 子序列覆盖（字符缺失或合并时的“包含关系”）
fn composite_similarity(a: &str, b: &str) -> f32 {
    let lev = similarity_ratio(a, b);
    let lcs = lcs_ratio(a, b);
    lev.max(lcs)
}

fn similarity_ratio(a: &str, b: &str) -> f32 {
    let max_len = a.chars().count().max(b.chars().count()) as f32;
    if max_len == 0.0 {
        return 1.0;
    }

    let dist = levenshtein(a, b) as f32;
    (1.0 - dist / max_len).max(0.0)
}

/// 最长公共子序列（LCS）长度占较长串长度的比例。
fn lcs_ratio(a: &str, b: &str) -> f32 {
    let len_a = a.chars().count();
    let len_b = b.chars().count();
    let max_len = len_a.max(len_b) as f32;
    if max_len == 0.0 {
        return 1.0;
    }

    let lcs_len = lcs_len(a, b) as f32;
    (lcs_len / max_len).max(0.0)
}

fn normalize(s: &str, case_sensitive: bool) -> String {
    // 归一化：
    // - NFKC 将全角/半角、兼容形归一
    // - 去除所有空白（中文场景常有多余空格）
    // - 小写化（若非区分大小写）
    let mut normalized: String = s.nfkc().collect();
    normalized.retain(|c| !c.is_whitespace());
    if !case_sensitive {
        normalized = normalized.to_lowercase();
    }
    normalized
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();

    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }

    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr: Vec<usize> = vec![0; b.len() + 1];

    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        prev.clone_from(&curr);
    }

    prev[b.len()]
}

/// 计算最长公共子序列长度，O(n*m) 动态规划，适用于短文本匹配。
fn lcs_len(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut dp = vec![vec![0; b.len() + 1]; a.len() + 1];
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            if a[i - 1] == b[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }
    dp[a.len()][b.len()]
}
