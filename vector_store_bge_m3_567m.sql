CREATE TABLE vector_store_bge_m3_567m (
    id           BIGSERIAL PRIMARY KEY
  , doc_id       TEXT        NOT NULL
  , chunk_index  INT         NOT NULL
  , content      TEXT        NOT NULL
  , metadata     JSONB       NOT NULL DEFAULT '{}'::JSONB
  , embedding    VECTOR(1024) NOT NULL
  , updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
  , UNIQUE (doc_id, chunk_index)
);

CREATE INDEX ON vector_store_bge_m3_567m USING hnsw (embedding vector_cosine_ops);

COMMENT ON TABLE  vector_store_bge_m3_567m IS 'チャンク単位のベクトルストア。ベクトル算出モデル bge-m3:567m 専用';
COMMENT ON COLUMN vector_store_bge_m3_567m.id          IS '代理キー, 論理的な一意性は (doc_id, chunk_index) が担保する';
COMMENT ON COLUMN vector_store_bge_m3_567m.doc_id      IS '原典の識別子, URL やファイルパスなど。再取り込み時の削除・upsert のキー';
COMMENT ON COLUMN vector_store_bge_m3_567m.chunk_index IS '原典を分割した際の 0 起点の連番, 分割処理が採番する。前後チャンク取得の順序復元に使用';
COMMENT ON COLUMN vector_store_bge_m3_567m.content     IS 'チャンク本文, embedding の元テキストであり LLM へ渡す実体でもある';
COMMENT ON COLUMN vector_store_bge_m3_567m.metadata    IS 'チャンクに付随する属性情報, title, url, source, lang, fetched_at, section, tags, embed_model など。content を構造化したものではない';
COMMENT ON COLUMN vector_store_bge_m3_567m.embedding   IS 'content を bge-m3:567m でベクトル化したもの (1024次元), コサイン距離 <=> で検索する';
COMMENT ON COLUMN vector_store_bge_m3_567m.updated_at  IS '更新日時, upsert 時は明示的に now() を設定すること';