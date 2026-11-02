CREATE TABLE search_cache_bge_m3_567m (
    cache_key             TEXT NOT NULL
  , search_method_version TEXT NOT NULL
  , query_text            TEXT NOT NULL
  , query_embedding       vector(1024)
  , results               JSONB NOT NULL DEFAULT '[]'::jsonb
  , is_empty              BOOLEAN GENERATED ALWAYS AS (jsonb_array_length(results) = 0) STORED
  , source                TEXT
  , attempt_count         INT DEFAULT 1
  , searched_at           TIMESTAMPTZ NOT NULL DEFAULT now()
  , expires_at            TIMESTAMPTZ
  , PRIMARY KEY (cache_key, search_method_version)
);

CREATE INDEX ON search_cache_bge_m3_567m USING hnsw (query_embedding vector_cosine_ops)
  WHERE NOT is_empty;
CREATE INDEX ON search_cache_bge_m3_567m (searched_at DESC);

COMMENT ON TABLE  search_cache_bge_m3_567m IS 'クエリ単位の検索結果キャッシュ。ベクトル算出モデル bge-m3:567m 専用';
COMMENT ON COLUMN search_cache_bge_m3_567m.cache_key             IS 'キャッシュ検索用のキー, query_text を SHA256(HEX) でハッシュ化したもの';
COMMENT ON COLUMN search_cache_bge_m3_567m.search_method_version IS '検索設定組み合わせのバージョンキー, 複数バージョンの検索結果が混ざっては困るので区別する ... 正規化に使用する LLM モデル, 何個のキーワードを検索に用いるか, 正規化のロジック自体, ハッシュ計算方法, ベクトルの生成方法 など';
COMMENT ON COLUMN search_cache_bge_m3_567m.query_text            IS '正規化済み検索フレーズ, 抽出キーワードを昇順ソートし半角空白で連結したもの (正規化WFの hashBase)';
COMMENT ON COLUMN search_cache_bge_m3_567m.query_embedding       IS 'query_text を bge-m3:567m でベクトル化したもの, 初回登録時は NULL でミス後に遅延で埋める';
COMMENT ON COLUMN search_cache_bge_m3_567m.results               IS '検索結果, トップレベルは必ず JSON 配列。各要素は {content, doc_id, metadata} 形式';
COMMENT ON COLUMN search_cache_bge_m3_567m.is_empty              IS '検索結果が空の場合 true, results から自動算出される生成列。HNSW 部分インデックスの条件にも使用';
COMMENT ON COLUMN search_cache_bge_m3_567m.source                IS '検索結果の主な取得元, pgvector, searxng, tavily など';
COMMENT ON COLUMN search_cache_bge_m3_567m.attempt_count         IS '実検索を実行した累計回数, 空振りクエリの再検索抑制に使用 (キャッシュ参照回数ではない)';
COMMENT ON COLUMN search_cache_bge_m3_567m.searched_at           IS '最後に実検索を実行した日時, キャッシュヒット時は更新しない';
COMMENT ON COLUMN search_cache_bge_m3_567m.expires_at            IS 'キャッシュの期限切れ日時, NULL は無期限を意味する';